import numpy as np
import math

class Env(object):
    def __init__(self):
        self.n_cust=10
        self.depot=[]
        self.capacity = 15               #vehicle capacity
        self.input_dim = 3               #x,y,demand
        self.input_data = np.zeros((self.n_cust,self.input_dim))
        self.demand = self.input_data[:,-1]
        self.path=[self.depot]
        self.reward=0
        self.was_zero=False
        self.total_reward=0
        rnd=np.random
        x = rnd.uniform(1,100,size=(self.n_cust,2))
        d = rnd.randint(1,10,[self.n_cust,1])
        self.depot=[rnd.uniform(0,1),rnd.uniform(0,1)]
        self.input_data = np.concatenate([x,d],1)
        self.input_copy = self.input_data.copy()
        
    def create_dataset(self):
        self.input_data=self.input_copy.copy()
        #print(self.depot)
        #print(self.input_data)
        
    def reset(self):
        self.create_dataset()
        self.path=[self.depot]
        self.reward=0
        
    def update_state(self,cust_id):
        if self.input_data[cust_id,-1]==0:
            self.was_zero=True
            return
        else:
            self.was_zero=False
            if self.capacity>self.input_data[cust_id,-1]:
                self.capacity=self.capacity-self.input_data[cust_id,-1]
                self.input_data[cust_id,-1]=0
            else:
                self.input_data[cust_id,-1]=self.input_data[cust_id,-1]-self.capacity
                self.capacity=max(0,self.capacity-self.input_data[cust_id,-1])
            
    def is_over(self):
        #print(self.input_data[:,-1])
        return not self.input_data[:,-1].any()
    
    def observe(self):
        return self.input_data.copy(), self.path.copy()
    
    def act(self, cust_id):
        self.update_state(cust_id)
        if not self.was_zero:
            self.path.append(self.input_data[cust_id,:-1].tolist())
        reward = self.get_reward(cust_id)
        if self.capacity==0 and self.input_data[:,-1].any():
            self.path.append(self.depot)
            reward = self.get_reward(cust_id)
            self.capacity=15
        game_over = self.is_over()
        return self.input_data.copy(), self.path.copy(), reward, game_over
    
    def get_reward(self, cust_id):
        
        if self.was_zero:
            return 0
        
        dist = float(math.sqrt( ((self.path[-2][0]-self.path[-1][0])**2) + ((self.path[-2][1]-self.path[-1][1])**2) ))
        if self.is_over():
            dist=float(dist+math.sqrt( ((self.path[-2][0]-self.path[-1][0])**2) + ((self.path[-2][1]-self.path[-1][1])**2) )) 
        #self.reward=self.reward-dist*100
        return dist
