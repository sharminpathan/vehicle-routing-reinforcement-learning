import numpy as np
import random
import keras
from keras.optimizers import Adam
import tensorflow as tf
import collections
import math
import json
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, LSTM, Masking, concatenate

class Agent(object):
    def __init__(self, env, sess):
        self.env  = env
        self.sess = sess        
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        self.memory = list()
        self.num_actions=10
        
        self.actor_state_input, self.decoder_state_input, self.actor_model = self.create_actor_model()
        _, _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.num_actions]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        
        
        self.critic_actor_input, self.critic_decoder_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())
        
    
    def remember(self, states, done):
        self.memory.append([states, done])
        
    def create_actor_model(self):
        actor_in=Input(shape=(10,3)) 
        actor=Conv1D(128,1) (actor_in)
        actor=Flatten() (actor)
        actor_c=Dense(128) (actor)

        decoder_in=Input(shape=(None,2))
        decoder=Masking(mask_value=[0,0]) (decoder_in)
        decoder_out=LSTM(128, dropout=0.1) (decoder)

        actor=concatenate([actor_c, decoder_out])
        actor_critic_input=Activation('tanh') (actor)
        actor_out=Dense(self.num_actions, activation='softmax') (actor)
        
        model = keras.models.Model(input=[actor_in,decoder_in], output=actor_out)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return actor_in, decoder_in, model
    
    def create_critic_model(self):
        actor_in=Input(shape=(10,3)) 
        actor=Conv1D(128,1) (actor_in)
        actor=Flatten() (actor)
        actor_c=Dense(128) (actor)

        decoder_in=Input(shape=(None,2))
        decoder=Masking(mask_value=[0,0]) (decoder_in)
        decoder_out=LSTM(128, dropout=0.1) (decoder)

        actor=concatenate([actor_c, decoder_out])
        actor_critic_input=Activation('tanh') (actor)
        
        action_input=Input(shape=(10,))
        action_out=Dense(128)(action_input)
        
        merged=concatenate([actor_critic_input, action_out])
        merged_h1=Dense(128)(merged)
        critic_out=Dense(1)(merged_h1)
        model=keras.models.Model(input=[actor_in, decoder_in, action_input], output=critic_out)
        
        adam=Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return actor_in, decoder_in, action_input, model
    

    def _train_actor(self, samples):
        for sample in samples:
            input_state_t, path_state_t, action_t, reward_t, input_state_tp1, path_state_tp1 = sample[0]
            predicted_action = self.actor_model.predict([np.expand_dims(input_state_t, axis=0), np.expand_dims(path_state_t, axis=0)])
            state_input_t=[input_state_t, path_state_t]
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_actor_input:  np.expand_dims(input_state_t, axis=0),
                self.critic_decoder_input: np.expand_dims(path_state_t, axis=0),
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: np.expand_dims(input_state_t, axis=0),
                self.decoder_state_input: np.expand_dims(path_state_t, axis=0),
                self.actor_critic_grad: grads
            })
            
    def _train_critic(self, samples):
        for sample in samples:
            input_state_t, path_state_t, action_t, reward_t, input_state_tp1, path_state_tp1 = sample[0]
            done = sample[1]
            if not done:
                target_action = self.target_actor_model.predict([np.expand_dims(input_state_tp1, axis=0), np.expand_dims(path_state_tp1, axis=0)])[0]
                future_reward = self.target_critic_model.predict([np.expand_dims(input_state_tp1, axis=0), np.expand_dims(path_state_tp1, axis=1), np.expand_dims(target_action,axis=0)])[0][0]
                reward_t += self.gamma * future_reward
                
            self.critic_model.fit([np.expand_dims(input_state_tp1, axis=0), np.expand_dims(path_state_tp1, axis=0), np.expand_dims(action_t,axis=0)], [reward_t], verbose=0)
            

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, input_tm1, path_tm1):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action=np.random.randint(0, self.num_actions)
            act_array=np.zeros((self.num_actions))
            act_array[:]=0.1
            act_array[action]=1
        else: 
            act_array=self.actor_model.predict([np.expand_dims(input_tm1, axis=0), np.expand_dims(path_tm1, axis=1)])[0]
            
        while(input_tm1[np.argmax(act_array,-1]==0):
            act_array[np.argmax(act_array)]=0
                        
        return act_array
        
    
    
