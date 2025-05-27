import os
import torch
from torchvision.io import read_image
from torchvision.transforms import v2, InterpolationMode

img_dir = "data/train"
img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

mean = torch.zeros(3)
std = torch.zeros(3)
invalid_imgs = []
total_valid = 0

# Define resize transform
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32),
    v2.Resize((1850, 1850), interpolation=InterpolationMode.BICUBIC),
])

for f in img_files:
    try:
        # Read image and convert to float
        img = read_image(os.path.join(img_dir, f)).float() / 255.0
        
        # Handle grayscale images
        if img.shape[0] != 3:
            invalid_imgs.append(f)
            print(f"Image {f} has {img.shape[0]} channels, expected 3.")
            continue
            
        # Apply resize transform
        img = transform(img)
        
        # Calculate statistics on resized image
        mean += img.mean(dim=[1,2])
        std += img.std(dim=[1,2])
        total_valid += 1
        
    except Exception as e:
        print(f"Error reading {f}: {e}")

mean /= total_valid
std /= total_valid

print("\nStatistics after resize to 1850x1850:")
print("Mean:", mean.tolist())
print("Std:", std.tolist())
print(f"Total valid images processed: {total_valid}")
print("Images with channels != 3:", len(invalid_imgs))

# Save the computed values
with open('image_stats.txt', 'w') as f:
    f.write(f"Mean: {mean.tolist()}\n")
    f.write(f"Std: {std.tolist()}\n")
    f.write(f"Total valid images: {total_valid}\n")
    f.write(f"Invalid images: {len(invalid_imgs)}\n")

"""
Without transform

TRAIN
Mean: [0.34289437532424927, 0.3486197590827942, 0.31862545013427734]
Std: [0.16172315180301666, 0.1536036878824234, 0.1492483913898468]
Images with channels != 3 (166 imgs - 1d gray scale): ['P1765.png', 'P1656.png', 'P1201.png', 'P1368.png', 'P1517.png', 'P1730.png', 'P1702.png', 'P1727.png', 'P1556.png', 'P1317.png', 'P1279.png', 'P1534.png', 'P1364.png', 'P1599.png', 'P1250.png', 'P1200.png', 'P1337.png', 'P1305.png', 'P1608.png', 'P1336.png', 'P1673.png', 'P1552.png', 'P1662.png', 'P1768.png', 'P1259.png', 'P1350.png', 'P1249.png', 'P1526.png', 'P1753.png', 'P1524.png', 'P1256.png', 'P1705.png', 'P1245.png', 'P1587.png', 'P1582.png', 'P1514.png', 'P1615.png', 'P1354.png', 'P1329.png', 'P1696.png', 'P1691.png', 'P1744.png', 'P1688.png', 'P1360.png', 'P1243.png', 'P1740.png', 'P1224.png', 'P1338.png', 'P1600.png', 'P1707.png', 'P1298.png', 'P1571.png', 'P1545.png', 'P1585.png', 'P1361.png', 'P1339.png', 'P1357.png', 'P1616.png', 'P1292.png', 'P1658.png', 'P1322.png', 'P1717.png', 'P1277.png', 'P1274.png', 'P1208.png', 'P1748.png', 'P1682.png', 'P1539.png', 'P1353.png', 'P1255.png', 'P1238.png', 'P1597.png', 'P1670.png', 'P1216.png', 'P1217.png', 'P1251.png', 'P1521.png', 'P1300.png', 'P1555.png', 'P1207.png', 'P1540.png', 'P1276.png', 'P1639.png', 'P1351.png', 'P1580.png', 'P1232.png', 'P1272.png', 'P1515.png', 'P1638.png', 'P1652.png', 'P1607.png', 'P1291.png', 'P1228.png', 'P1669.png', 'P1366.png', 'P1683.png', 'P1726.png', 'P1365.png', 'P1640.png', 'P1602.png', 'P1680.png', 'P1609.png', 'P1685.png', 'P1281.png', 'P1554.png', 'P1537.png', 'P1674.png', 'P1589.png', 'P1641.png', 'P1620.png', 'P1676.png', 'P1341.png', 'P1535.png', 'P1675.png', 'P1614.png', 'P1308.png', 'P2686.png', 'P1205.png', 'P1646.png', 'P1563.png', 'P1631.png', 'P1659.png', 'P1756.png', 'P1618.png', 'P1531.png', 'P1689.png', 'P1591.png', 'P1295.png', 'P1558.png', 'P1299.png', 'P1562.png', 'P1757.png', 'P1660.png', 'P1369.png', 'P1746.png', 'P1241.png', 'P1211.png', 'P1741.png', 'P1222.png', 'P1265.png', 'P1747.png', 'P1214.png', 'P1221.png', 'P1743.png', 'P1703.png', 'P1724.png', 'P1325.png', 'P1297.png', 'P1321.png', 'P1649.png', 'P1519.png', 'P1533.png', 'P1565.png', 'P1661.png', 'P1736.png', 'P1709.png', 'P1333.png', 'P1309.png', 'P1617.png', 'P1698.png', 'P1260.png', 'P1725.png', 'P1633.png', 'P1653.png', 'P1358.png', 'P1632.png']


VAL
Mean: [0.3561701476573944, 0.3637610375881195, 0.3352423310279846]
Std: [0.16344082355499268, 0.15496723353862762, 0.15067626535892487]
Images with channels != 3 (37 imgs - 1 channel gray scale): ['P1522.png', 'P1551.png', 'P1708.png', 'P1288.png', 'P1697.png', 'P1267.png', 'P1307.png', 'P1319.png', 'P1714.png', 'P1606.png', 'P1686.png', 'P1343.png', 'P1328.png', 'P1287.png', 'P1209.png', 'P1306.png', 'P1574.png', 'P1636.png', 'P1340.png', 'P1258.png', 'P1239.png', 'P1719.png', 'P1344.png', 'P1223.png', 'P1311.png', 'P1758.png', 'P1645.png', 'P1359.png', 'P1704.png', 'P1745.png', 'P1550.png', 'P1231.png', 'P1352.png', 'P1594.png', 'P1247.png', 'P1739.png', 'P1634.png']


With transform (on txt)
"""