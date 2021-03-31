import run

if __name__ =='__main__':
    for x,i in enumerate([0.001,0.0001,0.01,0.1,0.00001]):
        for y,j in enumerate([16,32,1,64]):
            print("RUNNING ON ITERATION",x,y)
            run.RUN(i,j)