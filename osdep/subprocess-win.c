/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <windows.h>
#include <string.h>

#include "osdep/subprocess.h"

#include "osdep/io.h"
#include "osdep/atomics.h"

#include "talloc.h"
#include "common/common.h"
#include "stream/stream.h"
#include "misc/bstr.h"

static void write_arg(bstr *cmdline, char *arg)
{
    // If the string doesn't have characters that need to be escaped, it's best
    // to leave it alone for the sake of Windows programs that don't process
    // quoted args correctly.
    if (!strpbrk(arg, " \t\"")) {
        bstr_xappend(NULL, cmdline, bstr0(arg));
        return;
    }

    // If there are characters that need to be escaped, write a quoted string
    bstr_xappend(NULL, cmdline, bstr0("\""));

    // Escape the argument. To match the behavior of CommandLineToArgvW,
    // backslashes are only escaped if they appear before a quote or the end of
    // the string.
    int num_slashes = 0;
    for (int pos = 0; arg[pos]; pos++) {
        switch (arg[pos]) {
        case '\\':
            // Count backslashes that appear in a row
            num_slashes++;
            break;
        case '"':
            bstr_xappend(NULL, cmdline, (struct bstr){arg, pos});

            // Double preceding slashes
            for (int i = 0; i < num_slashes; i++)
                bstr_xappend(NULL, cmdline, bstr0("\\"));

            // Escape the following quote
            bstr_xappend(NULL, cmdline, bstr0("\\"));

            arg += pos;
            pos = 0;
            num_slashes = 0;
            break;
        default:
            num_slashes = 0;
        }
    }

    // Write the rest of the argument
    bstr_xappend(NULL, cmdline, bstr0(arg));

    // Double slashes that appear at the end of the string
    for (int i = 0; i < num_slashes; i++)
        bstr_xappend(NULL, cmdline, bstr0("\\"));

    bstr_xappend(NULL, cmdline, bstr0("\""));
}

// Convert an array of arguments to a properly escaped command-line string
static wchar_t *write_cmdline(void *ctx, char **argv)
{
    // argv[0] should always be quoted. Otherwise, arguments may be interpreted
    // as part of the program name. Also, it can't contain escape sequences.
    bstr cmdline = {0};
    bstr_xappend_asprintf(NULL, &cmdline, "\"%s\"", argv[0]);

    for (int i = 1; argv[i]; i++) {
        bstr_xappend(NULL, &cmdline, bstr0(" "));
        write_arg(&cmdline, argv[i]);
    }

    wchar_t *wcmdline = mp_from_utf8(ctx, cmdline.start);
    talloc_free(cmdline.start);
    return wcmdline;
}

static int create_overlapped_pipe(HANDLE *read, HANDLE *write)
{
    static atomic_ulong counter = ATOMIC_VAR_INIT(0);

    // Generate pipe name
    unsigned long id = atomic_fetch_add(&counter, 1);
    unsigned pid = GetCurrentProcessId();
    wchar_t buf[36];
    swprintf(buf, sizeof(buf), L"\\\\?\\pipe\\mpv-anon-%08x-%08lx", pid, id);

    // The function for creating anonymous pipes (CreatePipe) can't create
    // overlapped pipes, so instead, use a named pipe with a unique name
    *read = CreateNamedPipeW(buf, PIPE_ACCESS_INBOUND |
        FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_BYTE | PIPE_WAIT, 1, 0, 4096, 0, NULL);
    if (*read == INVALID_HANDLE_VALUE)
        goto error;

    // Open the write end of the pipe as a synchronous handle
    *write = CreateFileW(buf, GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (*write == INVALID_HANDLE_VALUE)
        goto error;

    return 0;
error:
    *read = *write = INVALID_HANDLE_VALUE;
    return -1;
}

static void delete_handle_list(void *p)
{
    LPPROC_THREAD_ATTRIBUTE_LIST list = p;
    VOID (WINAPI *pDeleteProcThreadAttributeList)(LPPROC_THREAD_ATTRIBUTE_LIST);

    HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    pDeleteProcThreadAttributeList =
        (VOID (WINAPI*)(LPPROC_THREAD_ATTRIBUTE_LIST))
        GetProcAddress(kernel32, "DeleteProcThreadAttributeList");

    if (pDeleteProcThreadAttributeList)
        pDeleteProcThreadAttributeList(list);
}

// Create a PROC_THREAD_ATTRIBUTE_LIST that specifies exactly which handles are
// inherited by the subprocess
static LPPROC_THREAD_ATTRIBUTE_LIST create_handle_list(void *ctx,
                                                       HANDLE *handles, int num)
{
    WINBOOL (WINAPI *pInitializeProcThreadAttributeList)(
            LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD, PSIZE_T);
    WINBOOL (WINAPI *pUpdateProcThreadAttribute)(LPPROC_THREAD_ATTRIBUTE_LIST,
            DWORD, DWORD_PTR, PVOID, SIZE_T, PVOID, PSIZE_T);

    // Load Windows Vista functions, if available
    HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    pInitializeProcThreadAttributeList =
        (WINBOOL (WINAPI*)(LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD, PSIZE_T))
        GetProcAddress(kernel32, "InitializeProcThreadAttributeList");
    pUpdateProcThreadAttribute =
        (WINBOOL (WINAPI*)(LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD_PTR,
                           PVOID, SIZE_T, PVOID, PSIZE_T))
        GetProcAddress(kernel32, "UpdateProcThreadAttribute");
    if (!pInitializeProcThreadAttributeList || !pUpdateProcThreadAttribute)
        return NULL;

    // Get required attribute list size
    SIZE_T size = 0;
    if (!pInitializeProcThreadAttributeList(NULL, 1, 0, &size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
            return NULL;
    }

    // Allocate attribute list
    LPPROC_THREAD_ATTRIBUTE_LIST list = talloc_size(ctx, size);
    if (!pInitializeProcThreadAttributeList(list, 1, 0, &size))
        goto error;
    talloc_set_destructor(list, delete_handle_list);

    if (!pUpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                    handles, num * sizeof(HANDLE), NULL, NULL))
        goto error;

    return list;
error:
    talloc_free(list);
    return NULL;
}

// Helper method similar to sparse_poll, skips NULL handles
static int sparse_wait(HANDLE *handles, unsigned num_handles)
{
    unsigned w_num_handles = 0;
    HANDLE w_handles[num_handles];
    int map[num_handles];

    for (unsigned i = 0; i < num_handles; i++) {
        if (!handles[i])
            continue;

        w_handles[w_num_handles] = handles[i];
        map[w_num_handles] = i;
        w_num_handles++;
    }

    if (w_num_handles == 0)
        return -1;
    DWORD i = WaitForMultipleObjects(w_num_handles, w_handles, FALSE, INFINITE);
    i -= WAIT_OBJECT_0;

    if (i >= w_num_handles)
        return -1;
    return map[i];
}

// Wrapper for ReadFile that treats ERROR_IO_PENDING as success
static int async_read(HANDLE file, void *buf, unsigned size, OVERLAPPED* ol)
{
    if (!ReadFile(file, buf, size, NULL, ol))
        return (GetLastError() == ERROR_IO_PENDING) ? 0 : -1;
    return 0;
}

static void write_none(void *ctx, char *data, size_t size)
{
}

int mp_subprocess(char **args, struct mp_cancel *cancel, void *ctx,
                  subprocess_read_cb on_stdout, subprocess_read_cb on_stderr,
                  char **error)
{
    wchar_t *tmp = talloc_new(NULL);
    int status = -1;
    struct {
        HANDLE read;
        HANDLE write;
        OVERLAPPED ol;
        char buf[4096];
        subprocess_read_cb read_cb;
    } pipes[2] = {
        { .read_cb = on_stdout ? on_stdout : write_none },
        { .read_cb = on_stderr ? on_stderr : write_none },
    };

    // If the function exits before CreateProcess, there was an init error
    *error = "init";

    for (int i = 0; i < 2; i++) {
        pipes[i].ol.hEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (!pipes[i].ol.hEvent)
            goto done;
        if (create_overlapped_pipe(&pipes[i].read, &pipes[i].write))
            goto done;
        if (!SetHandleInformation(pipes[i].write, HANDLE_FLAG_INHERIT,
                                                  HANDLE_FLAG_INHERIT))
            goto done;
    }

    // Convert the args array to a UTF-16 Windows command-line string
    wchar_t *cmdline = write_cmdline(tmp, args);

    DWORD flags = CREATE_UNICODE_ENVIRONMENT;
    PROCESS_INFORMATION pi = {0};
    STARTUPINFOEXW si = {
        .StartupInfo = {
            .cb = sizeof(si),
            .dwFlags = STARTF_USESTDHANDLES,
            .hStdInput = NULL,
            .hStdOutput = pipes[0].write,
            .hStdError = pipes[1].write,
        },

        // Specify which handles are inherited by the subprocess. If this isn't
        // specified, the subprocess inherits all inheritable handles, which
        // could include handles created by other threads. See:
        // http://blogs.msdn.com/b/oldnewthing/archive/2011/12/16/10248328.aspx
        .lpAttributeList = create_handle_list(tmp,
                (HANDLE[]){ pipes[0].write, pipes[1].write }, 2),
    };

    // PROC_THREAD_ATTRIBUTE_LISTs are only supported in Vista and up. If not
    // supported, create_handle_list will return NULL.
    if (si.lpAttributeList)
        flags |= EXTENDED_STARTUPINFO_PRESENT;

    // If we have a console, the subprocess will automatically attach to it so
    // it can receive Ctrl+C events. If we don't have a console, prevent the
    // subprocess from creating its own console window by specifying
    // CREATE_NO_WINDOW. GetConsoleCP() can be used to reliably determine if we
    // have a console or not (Cygwin uses it too.)
    if (!GetConsoleCP())
        flags |= CREATE_NO_WINDOW;

    if (!CreateProcessW(NULL, cmdline, NULL, NULL, TRUE, flags, NULL, NULL,
                        &si.StartupInfo, &pi))
        goto done;
    talloc_free(cmdline);
    talloc_free(si.lpAttributeList);
    CloseHandle(pi.hThread);

    // Init is finished
    *error = NULL;

    // List of handles to watch with sparse_wait
    HANDLE handles[] = { pipes[0].ol.hEvent, pipes[1].ol.hEvent, pi.hProcess,
                         cancel ? mp_cancel_get_event(cancel) : NULL };

    for (int i = 0; i < 2; i++) {
        // Close our copy of the write end of the pipes
        CloseHandle(pipes[i].write);
        pipes[i].write = NULL;

        // Do the first read operation on each pipe
        if (async_read(pipes[i].read, pipes[i].buf, 4096, &pipes[i].ol)) {
            CloseHandle(pipes[i].read);
            handles[i] = pipes[i].read = NULL;
        }
    }

    DWORD r;
    DWORD exit_code;
    while (pipes[0].read || pipes[1].read || pi.hProcess) {
        int i = sparse_wait(handles, MP_ARRAY_SIZE(handles));
        switch (i) {
        case 0:
        case 1:
            // Complete the read operation on the pipe
            if (!GetOverlappedResult(pipes[i].read, &pipes[i].ol, &r, TRUE)) {
                CloseHandle(pipes[i].read);
                handles[i] = pipes[i].read = NULL;
                break;
            }

            pipes[i].read_cb(ctx, pipes[i].buf, r);

            // Begin the next read operation on the pipe
            if (async_read(pipes[i].read, pipes[i].buf, 4096, &pipes[i].ol)) {
                CloseHandle(pipes[i].read);
                handles[i] = pipes[i].read = NULL;
            }

            break;
        case 2:
            GetExitCodeProcess(pi.hProcess, &exit_code);
            status = exit_code;

            CloseHandle(pi.hProcess);
            handles[i] = pi.hProcess = NULL;
            break;
        case 3:
            if (pi.hProcess) {
                TerminateProcess(pi.hProcess, 1);
                *error = "killed";
                goto done;
            }
            break;
        default:
            goto done;
        }
    }

done:
    for (int i = 0; i < 2; i++) {
        if (pipes[i].read) {
            // Cancel any pending I/O (if the process was killed)
            CancelIo(pipes[i].read);
            GetOverlappedResult(pipes[i].read, &pipes[i].ol, &r, TRUE);
            CloseHandle(pipes[i].read);
        }
        if (pipes[i].write) CloseHandle(pipes[i].write);
        if (pipes[i].ol.hEvent) CloseHandle(pipes[i].ol.hEvent);
    }
    if (pi.hProcess) CloseHandle(pi.hProcess);
    talloc_free(tmp);
    return status;
}
