# Flowr P2P Network Setup & Commands Guide

## 1. INSTALL FLOWR & DEPENDENCIES

### Option A: Using pip (Recommended)
```bash
# Step 1: Navigate to project directory
cd "d:\p2p network"

# Step 2: Create virtual environment (optional but recommended)
python -m venv venv

# Step 3: Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Step 4: Install all dependencies
pip install -r requirements.txt

# Alternative: Install individually
pip install p2pnetwork==1.4.6
pip install flwr==1.5.0
pip install numpy==1.24.3
```

### Option B: Using conda
```bash
conda create -n flowr-p2p python=3.10
conda activate flowr-p2p
pip install -r requirements.txt
```

---

## 2. VERIFY INSTALLATION

```bash
# Check if all packages installed correctly
pip list | grep -E "p2pnetwork|flwr|numpy"

# Or on Windows:
pip list | findstr "p2pnetwork flwr numpy"

# Test imports
python -c "import p2pnetwork; import flwr; print('✅ All packages installed')"
```

---

## 3. START P2P NODES

### Single Terminal (Sequential - NOT Recommended for testing)
```bash
python p2p_node.py 127.0.0.1 5000
```

### Multiple Terminals (RECOMMENDED)
Open **separate PowerShell/Terminal windows** for each node:

**Terminal 1: Node 1**
```bash
cd "d:\p2p network"
python p2p_node.py 127.0.0.1 5000
```

**Terminal 2: Node 2**
```bash
cd "d:\p2p network"
python p2p_node.py 127.0.0.1 5001
```

**Terminal 3: Node 3**
```bash
cd "d:\p2p network"
python p2p_node.py 127.0.0.1 5002
```

**Terminal 4+: Add more nodes as needed**
```bash
python p2p_node.py 127.0.0.1 5003
python p2p_node.py 127.0.0.1 5004
```

### Automated Node Startup (PowerShell Script)
Create `start_nodes.ps1`:
```powershell
# Start 3 nodes in background
$nodes = @(5000, 5001, 5002)

foreach ($port in $nodes) {
    Start-Process powershell -ArgumentList "-NoExit -Command python p2p_node.py 127.0.0.1 $port"
    Start-Sleep -Seconds 2
}

Write-Host "✅ All nodes started"
```

Run it:
```bash
powershell -ExecutionPolicy Bypass -File start_nodes.ps1
```

---

## 4. CONNECT NODES TO EACH OTHER

Once nodes are running, in each terminal, connect to the first node:

**From Node 2 terminal:**
```
>> connect 127.0.0.1 5000
```

**From Node 3 terminal:**
```
>> connect 127.0.0.1 5000
```

Or cross-connect:
```
>> connect 127.0.0.1 5001
```

---

## 5. TEST FLOWR TASKS

### From any node terminal:

**Test 1: Sum task**
```
>> task sum [10,20,30]
📋 Task created: abc-def-123
>> result abc-def-123
✅ Result: {'status': 'success', 'value': 60}
```

**Test 2: Ping task**
```
>> task ping
📋 Task created: xyz-789
>> result xyz-789
✅ Result: {'status': 'success', 'message': 'pong'}
```

**Test 3: Check task status**
```
>> status abc-def-123
📊 Task abc-def-123: completed
```

**Test 4: View all connected peers**
```
>> peers
Connected nodes:
   node-id-1  →  127.0.0.1:5000
   node-id-2  →  127.0.0.1:5001
```

---

## 6. SETUP NGINX (Load Balancing)

### Windows Installation

**Step 1: Download Nginx**
```bash
# Option A: Using Chocolatey
choco install nginx

# Option B: Manual download
# Go to http://nginx.org/en/download.html
# Download and extract to C:\nginx
```

**Step 2: Configure Nginx**
```bash
# Copy the provided nginx.conf to Nginx config directory
# Windows: C:\nginx\conf\nginx.conf
cp ".\nginx.conf" "C:\nginx\conf\nginx.conf"
```

**Step 3: Start Nginx**
```bash
# Navigate to Nginx directory
cd C:\nginx

# Start Nginx
.\nginx.exe

# Or install as service
.\nginx.exe -i
net start nginx

# Test if running
curl http://localhost:8080/health
```

**Step 4: Verify Load Balancer**
```bash
# Test with curl
curl http://localhost:8080

# Check Nginx status
curl http://localhost:8080/nginx_status
```

### Linux/Mac Installation

```bash
# Install Nginx
# Ubuntu/Debian:
sudo apt-get install nginx

# Mac:
brew install nginx

# Copy config
sudo cp nginx.conf /etc/nginx/sites-available/flowr-p2p

# Enable site
sudo ln -s /etc/nginx/sites-available/flowr-p2p /etc/nginx/sites-enabled/

# Test config
sudo nginx -t

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx  # Auto-start on reboot

# Verify
curl http://localhost:8080/health
```

---

## 7. NGINX LOAD BALANCING TEST

Once Nginx is running and nodes are connected:

**With Nginx Load Balancer:**
```bash
# Send request through Nginx (it routes to one of the nodes)
curl -X GET http://localhost:8080

# Check Nginx status
curl http://localhost:8080/nginx_status
```

**Monitor in Node Terminal:**
You'll see requests coming through Nginx:
```
✅ INBOUND connection from nginx-client
📨 Message from client: ...
```

---

## 8. NGINX CONFIGURATION TIPS

### Add More Nodes to Load Balancer
Edit `nginx.conf` upstream section:
```nginx
upstream flowr_nodes {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;  # Add new node
    server 127.0.0.1:5004;  # Add new node
}
```

Then reload:
```bash
# Windows
.\nginx.exe -s reload

# Linux/Mac
sudo systemctl reload nginx
```

### Load Balancing Strategies
Replace `upstream flowr_nodes` with:

**Least Connections (default):**
```nginx
upstream flowr_nodes {
    least_conn;
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
}
```

**IP Hash (sticky sessions):**
```nginx
upstream flowr_nodes {
    ip_hash;
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
}
```

**Random:**
```nginx
upstream flowr_nodes {
    random;
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
}
```

---

## 9. STOP NODES & NGINX

### Stop Nginx
```bash
# Windows
.\nginx.exe -s quit

# Linux/Mac
sudo systemctl stop nginx
```

### Stop Node Processes
```bash
# PowerShell - Kill all Python processes
Stop-Process -Name python -Force

# Command Prompt - Kill all Python processes
taskkill /IM python.exe /F

# Or manually close terminals (Ctrl+C)
```

---

## 10. QUICK START CHECKLIST

- [ ] Install Python 3.10+
- [ ] Run `pip install -r requirements.txt`
- [ ] Start Node 1: `python p2p_node.py 127.0.0.1 5000`
- [ ] Start Node 2: `python p2p_node.py 127.0.0.1 5001`
- [ ] Start Node 3: `python p2p_node.py 127.0.0.1 5002`
- [ ] Connect nodes: `connect 127.0.0.1 5000` (from other terminals)
- [ ] Test task: `task sum [1,2,3]`
- [ ] Install Nginx: `choco install nginx` (Windows)
- [ ] Copy `nginx.conf` to Nginx config directory
- [ ] Start Nginx: `.\nginx.exe` (Windows) or `sudo systemctl start nginx` (Linux)
- [ ] Test load balancer: `curl http://localhost:8080`

---

## 11. TROUBLESHOOTING

### Port Already in Use
```bash
# Find process using port 5000 (Windows)
netstat -ano | findstr :5000

# Kill process
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

### Nginx Won't Start
```bash
# Test config syntax
nginx -t

# Check if already running
# Windows: tasklist | findstr nginx
# Linux: ps aux | grep nginx
```

### Nodes Not Connecting
- Ensure firewall allows ports 5000-5004
- Check if nodes are on same network
- Verify IP address is correct (127.0.0.1 for localhost)

---

## 12. DIRECTORY STRUCTURE

```
d:\p2p network\
├── p2p_node.py              # Main P2P + Flowr integration
├── requirements.txt         # Python dependencies
├── nginx.conf              # Nginx load balancer config
├── start_nodes.ps1         # PowerShell script to start nodes
└── venv\                   # Virtual environment (optional)
```

---

Your setup is complete! You now have:
✅ Flowr Core orchestration
✅ P2P network with TCP transport
✅ Nginx load balancer for distributed requests
