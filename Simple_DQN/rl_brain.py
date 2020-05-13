import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1)

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay = 0.9,
            e_greedy = 0.9,
            replace_target_iter = 300,
            memory_size = 500,
            batch_size = 32,
            e_greedy_increment=None,
            output_graph = False,
                 ):
        self.n_actions = n_actions
        self.n_features= n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if  e_greedy_increment is not None else self.epsilon_max
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.loss_object = keras.losses.MSE
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size,n_features*2+2))
        self._build_net()
        # self.cost_his=[] Replaced By train_accuracy_results
        self.train_loss_results = []
        self.onfirst_update = True
        self.onfirst_run = True

    def _build_net(self):
        # ======================== Evaluation Neural Network =============================
        self.eval_layers = []
        k_init = tf.random_normal_initializer(0.,0.3)
        b_init = tf.constant_initializer(0.1)
        self.eval_layers.append(keras.layers.Dense(units=10,
                                                   activation=keras.activations.relu,
                                                   kernel_initializer=k_init,
                                                   bias_initializer=b_init))
        self.eval_layers.append(keras.layers.Dense(units=10,activation=keras.activations.relu,
                                                   kernel_initializer=k_init,
                                                   bias_initializer=b_init))
        self.eval_layers.append(keras.layers.Dense(units=self.n_actions,
                                                   activation=keras.activations.linear,
                                                   kernel_initializer=k_init,
                                                   bias_initializer=b_init))
        self.eval_net = keras.models.Sequential(self.eval_layers)

        #=======================Target Neural Network ===================================
        self.target_layers = []
        self.target_layers.append(keras.layers.Dense(units=10,
                                                   activation=keras.activations.relu,
                                                   kernel_initializer=k_init,
                                                   bias_initializer=b_init))
        self.target_layers.append(keras.layers.Dense(units=10, activation=keras.activations.relu,
                                                   kernel_initializer=k_init,
                                                   bias_initializer=b_init))
        self.target_layers.append(keras.layers.Dense(units=self.n_actions,
                                                   activation=keras.activations.linear,
                                                   kernel_initializer=k_init,
                                                   bias_initializer=b_init))
        self.target_net = keras.models.Sequential(self.target_layers)


    def loss(self,eval_x,target_y):
        eval_y = self.eval_net(eval_x)
        return self.loss_object(y_true = target_y,y_pred = eval_y)

    def grad(self,model,inputs,targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs,targets)
        return loss_value, tape.gradient(loss_value,model.trainable_variables)

    def train(self,model,input,output,learning_rate,loss,optimizer):# For single data
        loss_value,grads = self.grad(model,input,output)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        return loss_value

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,"memory_counter"):
            self.memory_counter = 0
        transition = np.hstack((s,[a,r],s_))
        # Replace the old memory with new memory Cicular queue
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:] # How to dock tensor between numpy8 array
        if np.random.uniform() < self.epsilon:
            if self.onfirst_run:
                self.observation = observation
                self.onfirst_run = False
            actions_value = self.eval_net.predict(observation) # TODO: Debug
            action = np.argmax(actions_value) # TODO: INSPECT INPUT and OUTPUT format
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            weights = self.eval_net.get_weights()
            if self.onfirst_update:
                self.target_net.predict(self.observation)
                self.onfirst_update = False
            self.target_net.set_weights(weights)
            print("\ntarget+params_replaced\n")
        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]
        q_next = self.target_net.predict(batch_memory[:,-self.n_features:])
        q_eval = self.eval_net.predict(batch_memory[:,:self.n_features])
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_act_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]
        q_target[batch_index,eval_act_index] = reward + self.gamma*np.max(q_next,axis=1)
        loss = self.train(self.eval_net,batch_memory[:,:self.n_features],q_target,self.lr,self.loss,self.optimizer)
        self.train_loss_results.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.train_loss_results)),self.train_loss_results)
        plt.ylabel("Cost")
        plt.xlabel("training steps")
        plt.show()

    def save(self,episode):
        self.eval_net.save_weights('evals\weights_'+str(episode)+"eval")
        self.target_net_net.save_weights('target\weights_'+str(episode)+"eval")
        print("\nWeight has been saved\n")

    def load(self,evalname,targetname):
        self.eval_net.load_weights(evalname)
        self.target_net_net.load_weights(targetname)
        print("\nWeight has been Loaded\n")















