from threading import Thread 
import sys
import cv2
from queue import Queue 

class VideoStream:
	def __init__(self, src=0, name='Camera'):
		self.stream = cv2.VideoCapture(src)
		(self.ret, self.frame) = self.stream.read()
		self.stopped = False 
		self.name = name

	def start(self):
		thread = Thread(target=self.update, name=self.name, args=())
		thread.daemon = True 
		thread.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return

			(self.ret, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self): 
		self.stopped = True