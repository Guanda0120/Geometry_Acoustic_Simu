# Room Prepare
# Use RoomGeometry class
roomTest = RoomGeometry(room_1, material_abs_list, material_sca_list, material_ID)
absMatrix = roomTest.absorption_List()
scaMatrix = roomTest.scater_List()
cenPoint,norDric,planeD = roomTest.planeNormalDirec()

#Source Prepare
'''
    var in source object
    volume is the volume of the sound field
    location is the ndarray,[x,y,z]position of the source
    spl is the inite sound presure level of the 
'''
volume = 5
location = np.array([0,0,1.2])
startSpl = 90
endSpl = startSpl-65

singleSource = AcousticSource(volume,location,startSpl)
startPreasure = singleSource.spl2Preasure()
startTime = singleSource.initeTime()
rayDirec,raySource = singleSource.generateAcousticRay()


# Simulate Progress
maxSpl = np.max(singleSource.preasure2SPL(startPreasure))
iterNum = 0

direcList = []
sourceList = []
preasureList = []
timeList = []

#test 


while maxSpl>=endSpl:
    simuProgress = Simulation(norDric,room_1,rayDirec,raySource,absMatrix,scaMatrix)
    planeIndex,m,intersectionPoint = simuProgress.checkIntersection3D(norDric,rayDirec,raySource)
    tempPreasure,tempTime = simuProgress.propagateReduce(raySource,intersectionPoint,startPreasure,startTime)
    tempRayDirec,tempPreasureR = simuProgress.reflection(absMatrix,scaMatrix,planeIndex,rayDirec,norDric,tempPreasure)
    
    direcList.append(rayDirec)
    sourceList.append(raySource)
    preasureList.append(tempPreasureR)
    timeList.append(tempTime)
    
    rayDirec = tempRayDirec
    raySource = intersectionPoint
    startPreasure = tempPreasureR
    startTime = tempTime
        
    iterNum+=1
    maxSpl=np.max(simuProgress.preasure2SPL(startPreasure))
    print('In iter',iterNum,':' )
    print('The max SPL is:',round(maxSpl,2),'dB')
    print('===========================================================')

#=====================Reciver Defination and DSP================

r_1 = Reciver(np.array([2,2,1.5]),0.1)
direcListR,sourceListR,timeListR,preasureListR = r_1.reshapeTensor(direcList,sourceList,timeList,preasureList)
intDirec,intSour,intTime,intP,intScaler = r_1.checkIntersection(direcListR,sourceListR,timeListR,preasureListR)
time,spl = r_1.propagateReduce(intDirec,intSour,intTime,intP,intScaler)

time_1000 = time[:,3]
spl_1000 = spl[:,3]

timeSort_1000 = np.sort(time_1000)
indexTime = np.argsort(time_1000)
splSort_1000 = np.zeros((timeSort_1000.shape[0]))

for i in range(timeSort_1000.shape[0]):
    splSort_1000[indexTime[i]] = spl_1000[i]

signal_1 = AdaptiveDSP()
f_r,slope,rt_1000 = signal_1.linearRegresion(splSort_1000,timeSort_1000)
rt_b,rt_1000_1 = signal_1.backIntegral(splSort_1000,timeSort_1000,90)


# plot data
fig_1, ax_1 = plt.subplots()
fig_1.suptitle('Filtered 1k Hz Decay Curve')
l_1 ,= ax_1.plot(timeSort_1000, splSort_1000, linewidth=0.5)
l_2 ,= ax_1.plot(timeSort_1000, f_r, linewidth=2)
l_3 ,= ax_1.plot(timeSort_1000, rt_b , linewidth=2)
ax_1.set_xlabel('Time (s)')
ax_1.set_ylabel('Sound Preasure Level (dB)')
plt.legend(handles=[l_1,l_2,l_3],labels=['Test','LSQ_Fillter','BackInegral'],loc = 'best')
plt.show()
print('Use LSQ method 1k Hz Reverb Time is ',np.round(rt_1000,2),'s')
print('Use Back Integral method 1k Hz Reverb Time(T20) is ',np.round(rt_1000_1,2),'s')
print('Use Eyling Equation, 1k Hz Reverb Time is 0.49s')
fig_1.savefig('1kHz_Decay_Curve_LSQ.png', dpi=300)

