from matplotlib import pyplot as plt
# plt.title('Min iteration VS time', loc='left')
plt.xlabel('Time')
plt.ylabel('Iteration')

xlim = 0

lx = [x/1000 for x in range(-2000,3000)]
ly = [x ** 2 for x in lx]
plt.axis('off')
plt.plot(
    lx,
    ly,
    markersize=0,
    marker=None,
    color='black'
)
plt.show()
plt.close()