import asyncio
from concurrent.futures import ThreadPoolExecutor
import pygame
import pygame.locals
import requests
import time
import socket 
import sys
import RPi.GPIO as GPIO
import random
#import serial
pygame.init()

response = "NO"
index_img, p1_score , p2_score, db_p1_score, db_p2_score = 0, 0, 0, "0" ,"0"
#scoket listener from local host
class socket_conection:
  def __init__(self) -> None:
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.bind(('10.22.130.219', 5001))
    self.s.listen(1)
    self.flag = True

    self.s, sender_address = self.s.accept()
    print('Connection established with:', sender_address)

    self.size = self.width, self.height = 1280, 720

    self.set_screen()

  #def set_serial(self):
    #self.ser = serial.Serial('COM3', 9600)

  def get_scores(self):
    #print("Getting scores")
    cnt_p1 = float(requests.get('https://test-6801d-default-rtdb.firebaseio.com/p1.json').json())
    cnt_p2 = float(requests.get('https://test-6801d-default-rtdb.firebaseio.com/p2.json').json())
    return str(round(cnt_p1, 2)), str(round(cnt_p2, 2))
  
  def get_mediapipe_combinations(self):
    data = self.s.recv(1024).decode()
    #print ('Server response ', data)
    img_ind, p1_combination, p2_combination = data.split()
    return str(img_ind), float(p1_combination), float(p2_combination)
  
  def run_socket(self):
    global response, index_img, p1_score , p2_score, db_p1_score, db_p2_score
    while True:
      t_index_img, t_p1_score , t_p2_score = self.get_mediapipe_combinations()
      if t_index_img != index_img and self.flag == True:
        index_img = t_index_img
        p1_score = t_p1_score
        p2_score = t_p2_score
        self.flag = False
        print("New index")
        #serial.write(str(index_img + str(p1_score) + str(p2_score)).encode())

      db_p1_score, db_p2_score = self.get_scores()

      if (self.flag == True):
        self.s.sendall(str("YES").encode())
      else:
        self.s.sendall(str("NO").encode())
      time.sleep(0.01)

      #ser.write(str(response + str(p1_score) + str(p2_score)).encode())

  def set_screen(self):
    self.screen = pygame.display.set_mode(self.size)

    bg = pygame.image.load("disco.jpg")
    bg = pygame.transform.scale(bg, (self.size))

    self.org = pygame.Surface.copy(bg)


  def run_interface(self):
    global response, index_img, p1_score , p2_score, db_p1_score, db_p2_score

    time_val = 1
    v_max = 3.3
    factor = 10

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE =(255, 255, 255)
    GRAY = (192, 192, 192)
    BLACK = (0, 0, 0)

    pygame.font.init()
    font = pygame.font.SysFont(None, 32)

    player_one = "Score Player 1: "
    player_two = "Score Player 2: "

    # GPIO setup
    GPIO.setmode(GPIO.BCM)

    #leds setup for 2 players 4 leds modifiable intensity with pwm
    GPIO.setup(18, GPIO.OUT)
    GPIO.setup(14, GPIO.OUT)
    GPIO.setup(15, GPIO.OUT)
    GPIO.setup(23, GPIO.OUT)

    #Relevator setup
    GPIO.setup(2, GPIO.OUT)
    #print("Debug")
    # Stablish the pwm for the leds
    p1_pwm_0 = GPIO.PWM(18, 100)
    p1_pwm_1 = GPIO.PWM(14, 100)
    p2_pwm_0 = GPIO.PWM(15, 100)
    p2_pwm_1 = GPIO.PWM(23, 100)


    # Start the pwm
    p1_pwm_0.start(0)
    p1_pwm_1.start(0)
    p2_pwm_0.start(0)
    p2_pwm_1.start(0)

    def select_led(p1_id):
      if p1_id < 50:
        return font.render("COULD DO BETTER", True, RED), [0,100]
      elif p1_id < 75:
        return font.render("GOOD", True, GREEN), [100,100]
      elif p1_id <= 100:
        return font.render("PERFECT", True, BLUE), [100,0]
      else:
        return 0, 0

    def leds_updated(arr_led, isPlayerOne):
      arr_led[0] = 0 if arr_led[0] < 0 else arr_led[0]
      arr_led[1] = 0 if arr_led[1] < 0 else arr_led[1]
      if isPlayerOne:
        p1_pwm_0.ChangeDutyCycle(arr_led[0])
        p1_pwm_1.ChangeDutyCycle(arr_led[1])
      else:
        p2_pwm_0.ChangeDutyCycle(arr_led[0])
        p2_pwm_1.ChangeDutyCycle(arr_led[1])

    def reset_leds():
      p1_pwm_0.ChangeDutyCycle(0)
      p1_pwm_1.ChangeDutyCycle(0)
      p2_pwm_0.ChangeDutyCycle(0)
      p2_pwm_1.ChangeDutyCycle(0)
      
    songs_array=["Mi sexy Chambelán.wav", "Saturday Night.wav", "Thriller.wav", "Wannabe.wav", "suavemente.wav", "Tiempo De Vals.wav"]
    
    song_dict= {
      "Mi sexy Chambelán.wav"   : 164,
      "Saturday Night.wav"      : 220,
      "Thriller.wav"            : 313,
      "Wannabe.wav"             : 175,
      "suavemente.wav"          : 268,
      "Tiempo De Vals.wav"      : 249
    }
    # print("Before Loop")
    i = random.randint(0, len(songs_array)-1)
    selected_song = songs_array[i]
    song= pygame.mixer.Sound(selected_song)
    final_time=song_dict.get(selected_song)
    song.play() 
    while True:
      if (self.flag == True):
        continue
      # print("Game Logic")
      time_init = time.time()
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # print("Closing")
            sys.exit()
      # print ("Getting scores")

      p1_txt, p1_arr_led = select_led(p1_score)
      p2_txt, p2_arr_led = select_led(p2_score)
      player_one_score = font.render(player_one + db_p1_score, True, (255, 255, 255))
      temp_surface_p1 = pygame.Surface(player_one_score.get_size())
      temp_surface_p1.fill((192, 192, 192))
      temp_surface_p1.blit(player_one_score, (0, 0))
      player_two_score = font.render(player_two  + db_p2_score, True, (255, 255, 255))
      temp_surface_p2 = pygame.Surface(player_two_score.get_size())
      temp_surface_p2.fill((192, 192, 192))
      temp_surface_p2.blit(player_two_score, (0, 0))


      self.screen.fill(BLACK)
      bg = pygame.Surface.copy(self.org)
      bg_rect = bg.get_rect()

      bg.blit(temp_surface_p1, (int(self.width * 0.1), int(self.height * 0.1)))
      bg.blit(temp_surface_p2, (int(self.width * 0.7), int(self.height * 0.1)))

      if (index_img != "0"):
        it_img = pygame.image.load("stickman{}.jpg".format(index_img))
        it_img = pygame.transform.scale(it_img, (int(self.width * 0.2), int(self.height * 0.4)))
        bg.blit(p1_txt, (int(self.width * 0.1), int(self.height * 0.3)))
        bg.blit(p2_txt, (int(self.width * 0.7), int(self.height * 0.3)))
        bg.blit(it_img, (int(self.width * 0.4), int(self.height * 0.3)))

      self.screen.blit(bg, bg_rect)
      
      pygame.display.update()
      #print("Time: " + str(time.time() - time_init))

      
      #time.sleep(1)
      
      pwm_factor = 100 / factor # 10

      # print(time.time() - cur_time , time_val)
      cur_time = temp_time =  time.time()
      while time.time() - cur_time < time_val:
        if (time.time() - temp_time) > time_val/factor: #100ms
          temp_time = time.time()
          print (p1_arr_led, p2_arr_led)
          leds_updated(p1_arr_led, True)
          leds_updated(p2_arr_led, False) 
          p1_arr_led = [item - pwm_factor for item in p1_arr_led]
          p2_arr_led = [item - pwm_factor for item in p2_arr_led]
        print("In loop leds")
      
      self.flag = True
      print(time.time() - cur_time , time_val) #debe ser de 1 seg
      

      print("Flag on true")
      time.sleep(0.01)

      


if __name__ == '__main__':
  game_thread = socket_conection()
  executor = ThreadPoolExecutor(10)
  loop = asyncio.new_event_loop()
  game_interface = loop.run_in_executor(executor, game_thread.run_interface)
  game_socket = loop.run_in_executor(executor, game_thread.run_socket)
  loop.run_forever()
