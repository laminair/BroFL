# FLBench - Monitoring

We provide extensive monitoring for all servers and devices instantiated with FLBench. 

## Network monitoring
We use Scapy to isolate the traffic between client and server. 
Please make sure to provide appropriate rights to your python process so scapy can read from inet sockets.

Here's how you do it: 
```
setcap cap_net_raw=eip venv/bin/python3
```