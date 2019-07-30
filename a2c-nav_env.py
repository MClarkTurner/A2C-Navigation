import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

class Environment:
	class Point:
		def __init__(self, x, y, theta=0.0):
			self.x = x
			self.y = y
			self.theta = theta

	def __init__(self):
		self.action_space_low = [-.2, 0.0]
		self.action_space_high = [.2, 0.2]

	def dist_to_goal(self):
		return math.sqrt((self.agent.y - self.goal.y)**2 + (self.agent.x - self.goal.x)**2)

	def angle_to_goal(self):
		angle_to_goal = math.atan2(self.goal.y-self.agent.y, self.goal.x-self.agent.x) - self.agent.theta

		# make sure that angle is between pi and -pi
		if(angle_to_goal < -math.pi):
			return angle_to_goal + 2* math.pi
		if(angle_to_goal > math.pi):
			return angle_to_goal - 2* math.pi
		return angle_to_goal

	def get_state(self):
		return [self.dist_to_goal(), self.angle_to_goal()]

	def scale_state(self, state):
		dist = state[0]/self.dist_to_goal() * 2 -1
		angle = state[1]/math.pi
		return np.array([dist, angle]).reshape([-1, 2])

	def get_reward(self):
		# reward increases the closer to the goal you are (ie. closer that dist_to_goal == 0)
		dist_rew = 1 - (self.dist_to_goal()/self.start_distance)

		#reward increases the more you are facing the goal (ie. closer that angle_to_goal == 0)
		angle_rew = (math.pi - abs(self.angle_to_goal())) / math.pi

		return dist_rew + angle_rew
	
	def reset(self):
		self.agent = self.Point(0.0, 0.0, 0.0)
		self.goal = self.Point(5.0, 0.0)
		self.start_distance = self.dist_to_goal()

		return self.get_state()

	def step(self, action):
		direction, speed = action[0], action[1]

		self.agent.theta += direction

		self.agent.x += math.cos(self.agent.theta) * speed
		self.agent.y += math.sin(self.agent.theta) * speed

		return self.get_state(), self.get_reward()

class A2C:
	def __init__(self, env, lr=1e-4, ent_coef=.01, v_coef=.5, discount=0.95):
		self.env = env
		self.discount = discount
		
		##### PLACEHOLDERS #####

		self.placeholders = {
				"state": tf.placeholder(tf.float32, [None, 2], name="state_ph"),
				"action": tf.placeholder(tf.float32, [None, 2], name="action_ph"),
				"td_target": tf.placeholder(tf.float32, [None, 1], name="td_target_ph"),
				"reward": tf.placeholder(tf.float32, [None, 1], name="reward_ph")}

		##### MODEL #####

		top = self.placeholders["state"]
		
		with tf.variable_scope("critic"):
			dense1 = tf.layers.dense(inputs=top, units=400, activation=tf.nn.elu, name="dense_1")
			dense2 = tf.layers.dense(inputs=dense1, units=400, activation=tf.nn.elu, name="dense_2")

			#get state value
			self.v_out = tf.layers.dense(inputs=dense2, units=1, name="value_out")
		
		with tf.variable_scope("actor"):
			dense1 = tf.layers.dense(inputs=top, units=80, activation=tf.nn.elu, name="dense_1")
			dense2 = tf.layers.dense(inputs=dense1, units=80, activation=tf.nn.elu, name="dense_2")
		
			# mu and sigma are used to define the Normal Distribution
			mu = tf.layers.dense(inputs=dense2, units=2, name="mu", activation = tf.nn.tanh)
			sigma = tf.layers.dense(inputs=dense2, units=2, name="sigma", activation = tf.nn.softplus) 

		##### CHOOSE ACTION #####

		#select actions
		normal_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
		act_out = tf.reshape(normal_dist.sample(1), [-1, 2])
		self.act_out = tf.clip_by_value(act_out, self.env.action_space_low, self.env.action_space_high)

		##### OPTIMIZE NETWORK #####

		#value loss tensor
		value_loss = tf.losses.mean_squared_error(self.placeholders["reward"],self.v_out)

		#policy loss tensors
		log_prob = -normal_dist.log_prob(self.placeholders["action"])
		
		advantage = tf.stop_gradient(self.placeholders["td_target"] - self.v_out)
		entropy = normal_dist.entropy()

		policy_loss = tf.reduce_mean(log_prob * advantage + ent_coef * tf.reduce_mean(entropy))

		#total loss
		combined_loss = tf.math.add(value_loss * v_coef, policy_loss)

		#optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		self.train_op = optimizer.minimize( combined_loss )

		##### SESSION VARIABLES #####

		# model saver
		self.sess = tf.Session()

	def init(self):
		self.sess.run(tf.global_variables_initializer())

	def get_action(self, s):
		s = self.env.scale_state(s)

		return np.squeeze(self.sess.run([self.act_out], 
				feed_dict={	self.placeholders["state"]: s }))

	def get_value(self, s):
		s = self.env.scale_state(s)

		return self.sess.run([self.v_out], 
				feed_dict={	self.placeholders["state"]: s })

	def update(self, mb_state, mb_value , mb_action, mb_reward):
		s = []
		for i in range(len(mb_state)):
			s.append(self.env.scale_state(mb_state[i]))

		s = np.array(s).reshape([-1,2])
		v = np.array(mb_value).reshape([-1,1])
		a = np.array(mb_action).reshape([-1,2])
		r = np.array(mb_reward).reshape([-1, 1])

		#calculate discounted reward
		cummulative_reward, d_reward = 0, []
		for rew in r[::-1]:
			cummulative_reward = rew + self.discount * cummulative_reward
			d_reward.append(cummulative_reward)
		r = np.array(d_reward)[::-1]

		# generate TD target
		td_target = np.array(r + self.discount * v).reshape([-1, 1])

		# update the model
		feed_dict = {
					self.placeholders["state"]: s,
					self.placeholders["action"]: a,
					self.placeholders["td_target"]:td_target,
					self.placeholders["reward"]: r}
		_ = self.sess.run([self.train_op], feed_dict)


	def run(self, horizon, num_iter=1, train=False):
		rewards = []
		path = []

		for step in range(num_iter):
			#reset the environment/start the environment
			s = self.env.reset()
			cummulative_reward = 0

			mb_state, mb_value, mb_action, mb_reward = [],[],[],[]
			for t in range(horizon):

				a = self.get_action(s)
				
				# step environment using action
				next_state, r = self.env.step(a) 
				cummulative_reward = r + self.discount * cummulative_reward
				
				if(train):				
					# get the value of the new state
					value_of_next_state = self.get_value(next_state) 

					# store trajectory
					mb_state.append(s)
					mb_value.append( value_of_next_state[0] )
					mb_action.append(a)
					mb_reward.append(r)
				else:
					path.append((self.env.agent.x, self.env.agent.y))
					print("pos: {: 4.2f}, {: 4.2f}".format(self.env.agent.x, self.env.agent.y)), 
					print("| state: {: 4.2f}, {: 4.2f}".format(s[0], s[1])), 
					print("| action: {: 4.2f}, {: 4.2f}".format(a[0], a[1])),
					print("| reward: {: 4.2f}".format(r))

				s = next_state

			rewards.append(cummulative_reward)
			print("step:", step, "reward:", cummulative_reward)
			if(train):
				self.update(mb_state, mb_value , mb_action, mb_reward)
				mb_state, mb_value, mb_action, mb_reward = [],[],[],[]
					
		if(train):
			#draw the accumulated rewards
			rewards = np.array(rewards)
			plt.ylabel("Cummulative Reward")
			plt.xlabel("Episode")
			plt.plot(rewards)
		else:
			# draw the path the agent takes
			path = np.array(path)

			for i in range(0, len(path)):
				plt.plot(path[i:i+1, 0], path[i:i+1,1],'ro-')
			plt.plot([0., .0], [.0, .0],'go-')
			plt.plot([self.env.goal.x, self.env.goal.x], [self.env.goal.y, self.env.goal.y],'bo-')
			plt.xlabel("X position")
			plt.ylabel("Y position")
			
		plt.show()



if __name__ == '__main__':
	env = Environment()
	alg = A2C(env, lr=1e-6)

	num_iter = 5#1000
	horizon = 100

	alg.init()
	alg.run(horizon, num_iter=num_iter, train=True)
	alg.run(horizon)
	
