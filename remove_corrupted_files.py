import os

corrupted_list = ['D:/Social/Rural/Rural_1006.jpeg', 'D:/Social/Rural/Rural_1056.jpeg', 'D:/Social/Rural/Rural_1072.jpeg', 'D:/Social/Urban/Urban_1514.jpeg', 'D:/Social/Urban/Urban_1615.jpeg', 'D:/Social/Urban/Urban_1616.jpeg', 'D:/Social/Urban/Urban_1630.jpeg', 'D:/Social/Urban/Urban_1631.jpeg', 'D:/Social/Urban/Urban_1632.jpeg', 'D:/Social/Urban/Urban_1633.jpeg', 'D:/Social/Urban/Urban_1634.jpeg', 'D:/Social/Urban/Urban_1636.jpeg', 'D:/Social/Urban/Urban_1637.jpeg', 'D:/Social/Urban/Urban_1638.jpeg', 'D:/Social/Urban/Urban_1679.jpeg', 'D:/Social/Urban/Urban_1681.jpeg', 'D:/Social/Urban/Urban_907.jpeg']

for file in corrupted_list:
    os.remove(file)