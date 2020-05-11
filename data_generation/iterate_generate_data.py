import subprocess

retailers = ['A', 'B', 'C', 'D', 'E']
single_item = True

for retailer in retailers:
    for instance in range(1, 101):
        cmd = 'python.exe generate_data.py {} {} {}'.format(retailer, instance, single_item)
        print(cmd)
        subprocess.call(cmd)