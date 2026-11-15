"""Identity-checked local process ownership and child-tree termination."""

import psutil


def process_matches(pid, created):
    if not pid or not created:
        return False
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - created) < 0.1 and process.is_running()
    except psutil.Error:
        return False


def stop_tree(parent):
    """Terminate this process and its descendants, leaving ancestors alone."""
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in [*reversed(children), parent]:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([*children, parent], timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
