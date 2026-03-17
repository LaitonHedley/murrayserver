import numpy as np
import copy
class Environment(object):
  def __init__(self):
    #Initialize elements if the task including size of grid-world, number of agents, pixel values of agents, landmark, initial locations of agents and landmarks 
    self.GH = 3 #height of grid world
    self.GW = 4 #width of grid world
    self.AGENT = 240.0 #pixel values of agents
    self.TARGETS = [150.0] #pixel values of targets
    self.AGENTS_X = 0 #x-coordinates of initial locations of agents
    self.AGENTS_Y = self.GH-1  #y-coordinates of initial locations of agents
    self.OBSTACLE = 100.0 
    self.OBSTACLES_YX = [(self.GH-2,1)]
    self.TARGETS_YX = [(0,self.GW-1)] #locations of landmarks
    self.REWARDS = [1]
    self.ACTIONS = 4 # move up, down, left, righ
    
  def reset(self):
    # '''
    # Reset everything. 
    # '''
    self.s_t = np.zeros([self.GH,self.GW], dtype=np.float64) #create an array that represents states of the grid-world 
    # Agents and landmarks are initialised:
    # Agent x and y positions can be set in the following lists.
    self.agents_x = copy.deepcopy(self.AGENTS_X)
    self.agents_y = copy.deepcopy(self.AGENTS_Y)
    self.agent_status= False
    #Set pixel values of agents
    self.s_t[self.agents_y][self.agents_x] += self.AGENT
    #Initialize the landmarks in the environment
    for l,p in zip(self.TARGETS_YX,self.TARGETS):
      self.s_t[l[0]][l[1]] = p
    for yx in self.OBSTACLES_YX:
      self.s_t[yx[0]][yx[1]] = self.OBSTACLE 
    return self.s_t

  def step(self, action):
  # '''
  # Change environment state based on actions.
  # :param actions: List of integers providing actions for each agent
  # '''
    #Move agents according to actions.
    
    dx, dy = self.getDelta(action)
    targetx = self.agents_x + dx
    targety = self.agents_y + dy
    reward = 0 
    terminal = False 
    if self.noCollision(targetx, targety):
      for a, r in zip(self.TARGETS,self.REWARDS):
        if self.s_t[targety][targetx] == a:
          reward = r
          terminal = True 
          break
      self.s_t[self.agents_y][self.agents_x] -= self.AGENT
      self.s_t[targety][targetx] += self.AGENT
      self.agents_x = targetx
      self.agents_y = targety
      
    return self.s_t, reward, terminal 

  def getDelta(self, action):
    # '''
    # Determine the direction that the agent should take based upon the action selected. The actions are: 'Up':0, 'Right':1, 'Down':2, 'Left':3, :param action: int
    # '''
    if action == 0:
      return 0, -1
    elif action == 1:
      return 1, 0    
    elif action == 2:
      return 0, 1    
    elif action == 3:
      return -1, 0
    elif action == 4:
      return 0, 0 
  def noCollision(self, x, y):
    # '''
    # Checks if x, y coordinate is currently empty 
    # :param x: Int, x coordinate
    # :param y: Int, y coordinate
    # '''
    if x < 0 or x >= self.GW or\
      y < 0 or y >= self.GH or\
      self.s_t[y][x] == self.OBSTACLE:
      return False
    else:
      return True