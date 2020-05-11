import subprocess

retailers = ['A', 'B', 'C', 'D', 'E']
for retailer in retailers:
    for instance in range(1, 101):
        cmd = "python.exe order_fulfillment_single_item.py {} {}".format(retailer, instance)
        print(cmd)
        subprocess.call(cmd)