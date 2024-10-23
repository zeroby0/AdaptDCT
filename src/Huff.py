import heapq
from collections import Counter
from typing import Dict, List, Tuple

class HuffmanNode:
    def __init__(self, byte: int = None, freq: int = 0):
        self.byte = byte
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(byte_freq: Dict[int, int]) -> HuffmanNode:
    heap = [HuffmanNode(byte, freq) for byte, freq in byte_freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    
    return heap[0]

def generate_huffman_codes(root: HuffmanNode) -> Dict[int, str]:
    codes = {}
    
    def traverse(node: HuffmanNode, code: str):
        if node.byte is not None:
            codes[node.byte] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
    
    traverse(root, '')
    return codes

def huffman_encode(data: bytes) -> Tuple[bytes, Dict[int, str]]:
    byte_freq = Counter(data)
    root = build_huffman_tree(byte_freq)
    codes = generate_huffman_codes(root)
    
    encoded = ''.join(codes[byte] for byte in data)
    padding = 8 - (len(encoded) % 8)
    encoded += '0' * padding
    
    encoded_bytes = bytes(int(encoded[i:i+8], 2) for i in range(0, len(encoded), 8))
    return encoded_bytes, codes

def huffman_decode(encoded_bytes: bytes, codes: Dict[int, str]) -> bytes:
    reversed_codes = {code: byte for byte, code in codes.items()}
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    
    decoded = bytearray()
    current_code = ''
    for bit in bit_string:
        current_code += bit
        if current_code in reversed_codes:
            decoded.append(reversed_codes[current_code])
            current_code = ''
    
    return bytes(decoded)

# # Example usage
# data = b"Hello, World! This is a test string for Huffman coding."
# encoded, codes = huffman_encode(data)
# decoded = huffman_decode(encoded, codes)

# print(encoded)
# print(codes)

# print(f"Original size: {len(data)} bytes")
# print(f"Encoded size: {len(encoded)} bytes")
# print(f"Compression ratio: {len(encoded) / len(data):.2f}")
# print(f"Decoded data: {decoded.decode('utf-8')}")