# Stopping containers when `docker stop` is denied

On some environments (like GB10), containers started as root may refuse `docker stop` even with `sudo`. When this happens, follow these steps to force termination:

1. **Find the PID of the container process on the host:**
   ```bash
   docker inspect <container-id-or-name> --format '{{.State.Pid}}'
   ```

2. **Kill gracefully (SIGTERM):**
   ```bash
   sudo kill <pid>
   ```

3. **Force kill if it doesn't exit within ~10 seconds:**
   ```bash
   sudo kill -9 <pid>
   ```
