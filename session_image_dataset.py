import numpy as np
from scapy.all import rdpcap
from collections import defaultdict
from scapy.all import IP, TCP, UDP, Raw
from torch.utils.data import Dataset
from tqdm import tqdm

def group_packets_by_session(packet_list):
    sessions = defaultdict(list)
    
    for pkt in packet_list:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto  # 6 for TCP, 17 for UDP
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            else:
                # Skip ICMP or other protocols without ports
                continue
            
            if src_ip < dst_ip:
                session_key = (src_ip, src_port, dst_ip, dst_port, proto)
            else:
                session_key = (dst_ip, dst_port, src_ip, src_port, proto)
            
            sessions[session_key].append(pkt)

    for session_key in sessions:
        sessions[session_key].sort(key=lambda x: x.time)
            
    return sessions

def create_image_from_packet(packet, m):
    if not Raw in packet:
        return np.zeros((int(m ** 0.5), int(m ** 0.5)), dtype=np.uint8)
    
    payload = packet[Raw].load
    payload_m = payload[0:m] + bytes([0 for i in range(max(0, m - len(payload)))])
    img = np.frombuffer(payload_m, dtype=np.uint8).reshape(int(m ** 0.5), int(m ** 0.5)) 
    return img
    
def create_image_from_session(session, n, m):
    rn = int(n ** 0.5)
    rm = int(m ** 0.5)

    assert rn ** 2 == n, "n is not a perfect square"
    assert rm ** 2 == m, "m is not a perfect square"

    pkt_images = [create_image_from_packet(packet, m) for packet in session[0:n]] 
    padding_images = [np.zeros((rm, rm), dtype=np.uint8) for extra in range(max(0, n - len(session)))] # not enough packets in session for full image

    pkt_lens = [min(packet[IP].len - packet[IP].ihl * 4, 1500) for packet in session[0:n]] # capping at 1500 in case of corrupted length field
    padding_lens = [1501 for extra in range(max(0, n - len(session)))] # 1501 is greater than max possible packet length
    
    image = np.stack(pkt_images + padding_images).reshape(rn, rn, rm, rm).transpose(0, 2, 1, 3)
    lengths = np.array(pkt_lens + padding_lens)

    return image.reshape(rn * rm, rn * rm), lengths

class SessionImageDataset(Dataset):
    def __init__(self, path_to_pcaps: list, labels_of_pcap: list, n , m):
        super().__init__()
        assert len(path_to_pcaps) == len(labels_of_pcap), "length of labels and pcap paths should be same"

        self.images = []
        self.lens = []
        self.labels = []

        for idx in range(len(path_to_pcaps)):
            print(f"Loading {path_to_pcaps[idx]}")
            packets = rdpcap(path_to_pcaps[idx])
            print(f"Grouping packets by session")
            sessions = group_packets_by_session(packets)

            for session in tqdm(list(sessions.values()), desc=f"Creating session images"):
                img, len_array = create_image_from_session(session, n, m)
                self.images.append(img)
                self.lens.append(len_array)
                self.labels.append(labels_of_pcap[idx])
            
            print(f"Loaded {len(sessions)} sessions from {path_to_pcaps[idx]}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.lens[index], self.labels[index]