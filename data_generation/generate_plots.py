import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
''''
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]
ax.bar(langs,students)
plt.show()
'''
#plt.style.use('ggplot')

proportions = [0.058, 0.06, 0.173, 0.268,0.144,0.091, 0.206]
percentage = [prop*100 for prop in proportions]
zones = ['1-2', '3', '4', '5', '6', '7', '8']
pos = range(len(zones))
plt.bar(pos,  percentage, color='blue')
formatter = ticker.PercentFormatter(decimals=0)
plt.axes().yaxis.set_major_formatter(formatter)
plt.xlabel('Zones')
plt.ylabel("Percentage of Stores")
#plt.title("Percentage of stores in zones")

plt.xticks(pos, zones)
plt.savefig('shipping_zone_distribution.pdf')
#plt.show()