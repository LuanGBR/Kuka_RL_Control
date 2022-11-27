import numpy as np
from math import *
import matplotlib.pyplot as plt
import random
#from sympy import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

    
    
####################################################################################################################################



class CreateRobot:
    '''Classe que implementa os principais aspectos do robo: posicao do efetuador, dinamica do robo e verifica se o proximo estado é terminal'''
    def __init__(self, state, point):
        self.state = state
        #self.new_state = new_state
        self.point = point

    def position(self):
        '''Recebe os angulos das juntas do robo e as coordenadas do efetuador no seu sistema de coordenadas
        e retorna o ponto do efetuador nas coordenadas da base'''

        # Declaracao dos parametros D-H do KUKA KR16
        # angulos em graus e distancias em mm
        a0, a1, a2, a3, a4, a5, a6 = 0, 260, 680, 35, 0, 0, 0
        d0, d1, d2, d3, d4, d5, d6 = 0, 675, 0, 0, 670, 0, 115
        theta1, theta2, theta3, theta4, theta5, theta6 = self.state[0], self.state[1], self.state[2] - 90, self.state[3], self.state[4], self.state[5]
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = 0, -90, 0, -90, 90, -90, 0

        # Matrizes de transformacao
        M01T = np.matrix([[cos(theta1), -sin(theta1), 0, a0],
                          [sin(theta1) * cos(alpha0), cos(theta1) * cos(alpha0), -sin(alpha0), -sin(alpha0) * d1],
                          [sin(theta1) * sin(alpha0), cos(theta1) * sin(alpha0), cos(alpha0), cos(alpha0) * d1],
                          [0, 0, 0, 1]])

        M12T = np.matrix([[cos(theta2), -sin(theta2), 0, a1],
                          [sin(theta2) * cos(alpha1), cos(theta2) * cos(alpha1), -sin(alpha1), -sin(alpha1) * d2],
                          [sin(theta2) * sin(alpha1), cos(theta2) * sin(alpha1), cos(alpha1), cos(alpha1) * d2],
                          [0, 0, 0, 1]])


        M23T = np.matrix([[cos(theta3), -sin(theta3), 0, a2],
                          [sin(theta3) * cos(alpha2), cos(theta3) * cos(alpha2), -sin(alpha2), -sin(alpha2) * d3],
                          [sin(theta3) * sin(alpha2), cos(theta3) * sin(alpha2), cos(alpha2), cos(alpha2) * d3],
                          [0, 0, 0, 1]])

        M34T = np.matrix([[cos(theta4), -sin(theta4), 0, a3],
                          [sin(theta4) * cos(alpha3), cos(theta4) * cos(alpha3), -sin(alpha3), -sin(alpha3) * d4],
                          [sin(theta4) * sin(alpha3), cos(theta4) * sin(alpha3), cos(alpha3), cos(alpha3) * d4],
                          [0, 0, 0, 1]])

        M45T = np.matrix([[cos(theta5), -sin(theta5), 0, a4],
                          [sin(theta5) * cos(alpha4), cos(theta5) * cos(alpha4), -sin(alpha4), -sin(alpha4) * d5],
                          [sin(theta5) * sin(alpha4), cos(theta5) * sin(alpha4), cos(alpha4), cos(alpha4) * d5],
                          [0, 0, 0, 1]])

        M56T = np.matrix([[cos(theta6), -sin(theta6), 0, a5],
                          [sin(theta6) * cos(alpha5), cos(theta6) * cos(alpha5), -sin(alpha5), -sin(alpha5) * d6],
                          [sin(theta6) * sin(alpha5), cos(theta6) * sin(alpha5), cos(alpha5), cos(alpha5) * d6],
                          [0, 0, 0, 1]])

        # Matriz de transformacao entre base e efetuador
        # M06T = np.matmul(M01T*M12T*M23T*M34T*M45T*M56T)
        M1 = np.matmul(M01T, M12T)
        M2 = np.matmul(M1, M23T)
        M3 = np.matmul(M2, M34T)
        M4 = np.matmul(M3, M45T)
        M06T = np.matmul(M4, M56T)
        point_ = np.append(self.point, 1)
        p = np.transpose(point_)
        position = np.matmul(M06T, p)

        # calculo da posicao do efetuador nas coordenadas da base
        # position = np.matmul(M06T, np.transpose(p))

        return position


    def dynamics(self, a):
        '''Recebe o estado atual, a açao realizada e retorna o proximo estado'''
        delta_theta = 0.1
        q1 = self.state[0], q2 = self.state[1], q3 = self.state[2], q4 = self.state[3], q5 = self.state[4], q6 = self.state[5]
        a1 = a[0], a2 = a[1], a3 = a[2], a4 = a[3], a5 = a[4], a6 = a[5]

        q1_new = q1 + a1 * delta_theta
        q2_new = q2 + a2 * delta_theta
        q3_new = q3 + a3 * delta_theta
        q4_new = q4 + a4 * delta_theta
        q5_new = q5 + a5 * delta_theta
        q6_new = q6 + a6 * delta_theta
        self.state = [q1_new, q2_new, q3_new, q4_new, q5_new, q6_new]

        return self.state
        
        

    def terminal(self, setpoint):
        '''Verifica se um estado é terminal ou não'''
        done = False

        # Verifica se o limite das articulacoes foi ultrapassado
        if (self.state[0] < -185 or self.state[0] > 185):
            done = True
        elif (self.state[1] < -155 or self.state[1] > 35):
            done = True
        elif (self.state[2] < -130 or self.state[2] > 154):
            done = True
        elif (self.state[3] < -350 or self.state[3] > 350):
            done = True
        elif (self.state[4] < -130 or self.state[4] > 130):
            done = True
        elif (self.state[5] < -350 or self.state[5] > 350):
            done = True

        point = self.position()

        # Verifica se a cesta se aproximou muito do chão
        if point[2] <= 10:
            done = True

        dist_setpoint = sqrt(((point[0] - setpoint[0]) ** 2) + ((point[1] - setpoint[1]) ** 2) + ((point[2] - setpoint[2]) ** 2))

        # Verifica se o setpoint foi atingido
        if dist_setpoint <= 0.5:
            done = True
        return done
                


####################################################################################################################################



def reward(s_new, setpoint, point):
    '''Funçao que calcula o valor da recompensa associada ao novo estado apos a açao'''
    robo = CreateRobot(s_new, point)
    terminate = robo.terminal(aetpoint)
    BoundaryError = False
    GroundCollision = False
    ReachedSetPoint = False

    x, y, z = s[0], s[1], s[2]
    p = [x, y, z]
    x_new, y_new, z_new = s_new[0], s_new[1], s_new[2]
    p_new = [x_new, y_new, z_new]
    x_sp, y_sp, z_sp = setpoint[0], setpoint[1], setpoint[2]

    #Verifica se algum estado terminal foi alcancado e sua respectiva punicao, recompensa

    if terminate:
        if (s_new[0] < -185 or s_new[0] > 185):
            BoundaryError = True
        elif (s_new[1] < -155 or s_new[1] > 35):
            BoundaryError = True
        elif (s_new[2] < -130 or s_new[2] > 154):
            BoundaryError = True
        elif (s_new[3] < -350 or s_new[3] > 350):
            BoundaryError = True
        elif (s_new[4] < -130 or s_new[4] > 130):
            BoundaryError = True
        elif (s_new[5] < -350 or s_new[5] > 350):
            BoundaryError = True

        point = robo.position()
        dist_setpoint = sqrt(((point[0] - setpoint[0]) ** 2) + ((point[1] - setpoint[1]) ** 2) + ((point[2] - setpoint[2]) ** 2))

        # Verifica se a cesta se aproximou muito do chão
        if point[2] <= 10:
            GroundCollision = True

        # Verifica se o setpoint foi atingido
        if dist_setpoint <= 0.5:
            ReachedSetPoint = True

    if BoundaryError == False:
        boundaryPenalty = 0
    else:
        boundaryPenalty = -10

    if GroundCollision == False:
        collisionPenalty = 0
    else:
        collisionPenalty = -20

    if ReachedSetPoint == False:
        winBonus = 0
    else:
        winBonus = 20


    reward = 0.1*(-1.2 * (dist_setpoint))**2 + winBonus + boundaryPenalty + collisionPenalty
    #Verifica se o set point foi alcançado
    return reward

#z cima, y parede, x camera



####################################################################################################################################



class Ball:
    '''Classe que imlementa a cinematica da bola'''
    def __init__(self, position, velocity, timeDelta, numTrajPoints):
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.vx = velocity[0]
        self.vy = velocity[1]
        self.vz = velocity[2]
        self.timeDelta = timeDelta
        self.numTrajPoints = numTrajPoints
        self.g = -9.81

    def eulerExtrapolate(self):
        '''Metodo que calcula a proxima posicao e velocidade com base nas posicoes e velocidades iniciais
        utilizando o metodo de Euler'''
        self.x += self.vx * self.timeDelta
        self.y += self.vy * self.timeDelta
        self.z += self.vz * self.timeDelta

        self.vx += self.g * self.timeDelta
        self.vy += self.g * self.timeDelta
        self.vz += self.g * self.timeDelta

        position = [self.x, self.y, self.z]
        velocity = [self.vx, self.vy, self.vz]

        return (position, velocity)

    def getTrajectory(self):
        '''Funcao que estima a trajetoria da bola'''
        positions = []
        initialPosition = [self.x, self.y, self.z]
        initialVelocity = [self.vx, self.vy, self.vz]
        position = list(initialPosition)
        velocity = list(initialVelocity)
        for i in range(self.numTrajPoints):
            position, velocity = self.eulerExtrapolate()
            positions.append(position[:])
        return positions

    def get_setpoint(self):
        positions = self.getTrajectory()
        for i in range(len(positions)):
            if positions[i][2] <= 0.30:
                setpoint = positions[i]
        return setpoint
                        
        

####################################################################################################################################



class createActionSpace:
    '''Classe que implementa o espaço de açoes'''
    def __init__(self, len):
        self.len = len

    def createActionMatrix(self):
        '''Cria a matriz do espaço de açoes'''
        GL1 = np.array([1., 0., -1.])
        GL2 = np.array([1., 0., -1.])
        GL3 = np.array([1., 0., -1.])
        GL4 = np.array([1., 0., -1.])
        GL5 = np.array([1., 0., -1.])
        GL6 = np.array([1., 0., -1.])
        gl_6, gl_5, gl_4, gl_3, gl_2, gl_1 = np.meshgrid(GL1, GL2, GL3, GL4, GL5, GL6, indexing='ij')
        gl1 = np.reshape(gl_1, (729,))
        gl2 = np.reshape(gl_2, (729,))
        gl3 = np.reshape(gl_3, (729,))
        gl4 = np.reshape(gl_4, (729,))
        gl5 = np.reshape(gl_5, (729,))
        gl6 = np.reshape(gl_6, (729,))
        Auxiliar = np.array([gl_1, gl_2, gl_3, gl_4, gl_5, gl_6])
        Auxiliar2 = np.transpose(Auxiliar)
        #print("Auxiliar2 = ", Auxiliar2)
        actionSpace = np.zeros((729, 6), dtype=float)
        # actionSpace = np.zeros((243,5), dtype=float)
        actionSpace = np.reshape(Auxiliar2, (729, 6))
        return actionSpace

    def getSample(self):
        index = np.random.randint(0, self.len)
        actionSample = self.createActionMatrix()
        Sample = actionSample[index]
        return Sample
        
        
        
####################################################################################################################################



class DQN:
    def __init__(self, stateSpace, actionSpace):         #modificar env para dqn se tronar funçao do espaço de estados e do spaço de acoes
        self.stateSpace     = stateSpace
        self.actionSpace = actionSpace
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        
        self.model = self.create_model()
        self.target_model = self.create_model()
    
    def create_model(self):
        model   = Sequential()
        state_shape  = self.stateSpace.shape  #formato do ndarray do espaço de estados (12,)
        model.add(Dense(24, input_dim=state_shape[0], 
            activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.actionSpace.len))        #tamanho do espaço de acoes?
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.actionSpace.getSample()              #self.env.action_space.sample()       #pega uma amostra aleatoria do espaço de açoes
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
        
        
        
####################################################################################################################################



def main():
    #define parametros iniciais
    inicialpointBall = [2, 4, 2]
    inicialvelocity = [5, 5, 5]
    setpoint = ballExample.get_setpoint()
    stateSpace = np.array([0, 0, 0, 0, 0, 0])
    inicialpointRobot = [1, 0.5, 1.5]
    
    #cria espaço de açoes
    actionSpace = createActionSpace(729)
    AS = actionSpace.createActionMatrix()
    
    #cria bola
    ballExample = Ball(inicialpointBall, inicialvelocity, 2, 20)
    setpoint = ballExample.get_setpoint()
    
    #cria robo
    robot = CreateRobot(stateSpace, inicialpointRobot)
    
    #define parametros de aprendizado
    gamma = 0.9
    epsilon = .95
    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    #cria DQN
    dqn_agent = DQN(stateSpace, actionSpace)
    steps = []
    for trial in range(trials):
        cur_state = stateSpace        #stateSpace.reshape(6, 1)               #env.reset().reshape(1, 2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            #new_state = dynamics(cur_state, action)
            #new_state, reward, done, _ = env.step(action)
            new_state = robot.dynamics(action)
            p = robot.position()
            reward = reward(new_state, setpoint, p)
            done = terminal(new_state, setpoint)


            # reward = reward if not done else -20
            #new_state = new_state.reshape(6, 1)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break

if __name__ == "__main__":
    main()

'''
Funcionamento da main:
    - Cria objeto da classe Robo
    - Cria objeto da classe espaço de açoes
    - Cria objeto da classe DQN
    - No loop de treino chama a funçao action
    - Atualiza o estado do robo
    - Verifica se o novo estado é terminal
    - Guarda as informaçoes a respeito do estado atual, novo estado, açao, se o estado é terminal ou nao
    - Treina o agente
'''