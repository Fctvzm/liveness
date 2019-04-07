from threading import Thread 
import sys
import cv2
from queue import Queue 

SKIP = 5

class VideoStream:
	def __init__(self, path, queueSize=128):
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		count = 0
		while True:
			if self.stopped:
				return

			if not self.Q.full():
				grabbed, frame = self.stream.read()
				if not grabbed:
					self.stop()
					return
				if count % SKIP == 0:
					self.Q.put(frame)
				count += 1

	def read(self):
		return self.Q.get()

	def more(self):
		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True