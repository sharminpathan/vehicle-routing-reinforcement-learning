import numpy as np
import keras
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, SGD

import numpy as np
import pandas as pd
from math import*

from keras.models import Model, load_model
from keras.layers import merge, Activation, Conv1D, Dense, Embedding, Flatten, Input, LSTM, Masking, RepeatVector, Concatenate, concatenate, Multiply


EPISODES = 100

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 20
GAMMA = 0.99

BUFFER_SIZE = 1000
BATCH_SIZE = 100
NUM_ACTIONS = 3
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))



def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss
    
    
class Env:
    def __init__(self):
        self.capacity = 15
        self.path = [-1]
        self.depot = [0,0,0]
        self.orders_data = list()
        self.done = False
        self.mask = np.ones(NUM_ACTIONS)
        self.dist = 0
        self.total_dist = 0
        
    def reset(self):
        self.path = [-1]
        self.orders_data = np.array([[1,1,1],[2,2,2],[3,3,3]])
        self.done = False
        self.mask = np.ones(NUM_ACTIONS)
        self.dist = 0
        self.total_dist = 0
        
    def observe(self):
        return self.orders_data.copy()

    def is_over(self):
        return not self.orders_data[:,0].any()
    
    def get_reward(self, src, dest):
        if src==-1:
            return (abs(self.depot[1] - self.orders_data[dest,1]) + abs(self.depot[2] - self.orders_data[dest,2]))
        return (abs(self.orders_data[src,1] - self.orders_data[dest,1]) + abs(self.orders_data[src,2] - self.orders_data[dest,2]))
        
    def step(self, order_id):
        self.orders_data[order_id,0] = 0
        self.dist = -self.get_reward(self.path[-1],order_id)
        self.path.append(order_id)
        self.mask[order_id] = 0
        self.total_dist = self.total_dist + self.dist
        return self.orders_data.copy(), self.dist, self.is_over()
        
       
class Agent:
    def __init__(self):
        self.actor = self.actor_model()
        self.critic = self.critic_model()
        self.env = Env()
        self.test = 0
        self.reward = list()
            
    def actor_model(self):
        mask=Input(shape=(NUM_ACTIONS,), name='mask')
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))
        
        order_inp=Input(shape=(NUM_ACTIONS,3)) 
        order=Conv1D(128,1) (order_inp)
        order=Flatten() (order)
        order=Dense(128) (order)
        order=Dense(128) (order)
        split_actor=Dense(NUM_ACTIONS, activation='softmax') (order)
        split_actor_out=Multiply()([split_actor, mask])
        actor_model = keras.models.Model(input=[order_inp, mask, advantage, old_prediction], output=split_actor_out)
        
        adam=Adam(lr=LR)
        actor_model.compile(loss=[proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction)], optimizer=adam)
        return actor_model

    def critic_model(self):
        mask=Input(shape=(NUM_ACTIONS,), name='mask')
        order_inp=Input(shape=(NUM_ACTIONS,3)) 
        order=Conv1D(128,1) (order_inp)
        order=Flatten() (order)
        order=Dense(128) (order)
        order=Dense(128) (order)
        critic_out=Dense(1)(order)
        critic_model=keras.models.Model(input=[order_inp, mask], output=critic_out)
        
        adam=Adam(lr=LR)
        critic_model.compile(loss="mse", optimizer=adam)
        return critic_model
    
    def take_action(self):
        order_state = self.env.observe()
        p = self.actor.predict([np.expand_dims(order_state, axis=0), np.expand_dims(self.env.mask.copy(), axis=0), DUMMY_VALUE, DUMMY_ACTION])
        action=np.random.choice(np.where(self.env.mask==1)[0])
        act_array = np.zeros((NUM_ACTIONS))
        act_array[action] = 1
        return action, act_array, p

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA
            
    def get_batch(self):
        batch = [[], [], [], [], []]
        tmp_batch = [[], [], [], []]
        self.env.reset()
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.take_action()
            state = self.env.observe()
            mask = self.env.mask.copy()
            observation, reward, done = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(state)
            tmp_batch[1].append(mask)
            tmp_batch[2].append(action_matrix)
            tmp_batch[3].append(predicted_action)
            state = observation
            if done:
                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs, mask, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i], tmp_batch[3][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(mask)
                    batch[2].append(action)
                    batch[3].append(pred)
                    batch[4].append(r)
                tmp_batch = [[], [], [], []]
                self.reward = []
                self.env.reset()
        obs, mask, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.reshape(np.array(batch[4]), (len(batch[4]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, mask, action, pred, reward
    
    def predict(self):
        self.env.reset()
        score=0
        done=False
        while not done:
            state = self.env.observe()
            action = self.actor.predict([np.expand_dims(state, axis=0), np.expand_dims(self.env.mask.copy(), axis=0), DUMMY_VALUE, DUMMY_ACTION])
            if state[np.argmax(action),0]==0: 
                a = np.random.choice(np.where(self.env.mask==1)[0])
                action[a]=2
            state, reward, done = self.env.step(np.argmax(action))
            score+=reward
        print('path: ',self.env.path, 'total distance: ',np.abs(self.env.total_dist))
        
    def run(self):
        for i in range(EPISODES):
            obs, mask, action, pred, reward = self.get_batch()
            obs, mask, action, pred, reward = obs[:BUFFER_SIZE], mask[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict([obs, mask])

            advantage = reward - pred_values

            actor_loss = self.actor.fit([obs, mask, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss = self.critic.fit([obs, mask], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            
            self.predict()
            
            
if __name__ == '__main__':
    ag = Agent()
    ag.run()
