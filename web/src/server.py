#from typing_extensions import Required
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.renderers import render_to_response

import mysql.connector as mysql
import os

from flask import Flask, render_template
from PIL import Image
import base64
import io

db_user = os.environ['MYSQL_USER']
db_pass = os.environ['MYSQL_PASSWORD']
db_name = os.environ['MYSQL_DATABASE']
db_host = os.environ['MYSQL_HOST']

dirname = os.path.dirname(__file__)
image_path = os.path.join(dirname, 'public/unknown_faces')
# i_listing = os.listdir(image_path)
known_path = os.path.join(dirname, 'public/known_faces')
motion_path = os.path.join(dirname, 'public/motion_cap')
video_path = os.path.join(dirname, 'public/record_footage')
n = pid = 0


#path to home directory
def get_home(req):
  # Connect to the database and retrieve the users
  # db = mysql.connect(host=db_host, database=db_name, user=db_user, passwd=db_pass)
  # cursor = db.cursor()
  # cursor.execute("select first_name, last_name, email from Users;")
  # records = cursor.fetchall()
  # db.close()

  return render_to_response('templates/protecc.html', [], request=req)

#path to undetected people directory
def unknown_dir(req):
  l_img = get_latest_image(image_path)

  print(l_img)
  
  l_img = l_img.replace("public/", "")

  return render_to_response('templates/unknown_dir.html', {"img": l_img}, request=req)

#path to trusted people directory
def trust_dir(req):

  trusted_paths = get_all_image(known_path,"known_faces/")

  print(trusted_paths)

  return render_to_response('templates/trust_dir.html', {"img": trusted_paths}, request=req)

#path to videos directory
def record_dir(req):
  videos = get_all_video(video_path, "record_footage/")

  return render_to_response('templates/record_dir.html', {"vid": videos}, request=req)

#path to motion capture directory
def motion_dir(req):
  l_img = get_latest_image(motion_path)

  print(l_img)
  
  l_img = l_img.replace("public/", "")

  return render_to_response('templates/motion_dir.html', {"img": l_img}, request=req)

def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the latest image file in the given directory
    """

    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime)

def get_all_image(dirpath, pre, valid_extensions=('.jpg','.jpeg','.png')):
  images = list()


  for f in os.listdir(dirpath):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_extensions:
      continue
    images.append(os.path.join(pre,f))

  print(images)

  return images

def get_all_video(dirpath, pre, valid_extensions=('.mp4')):
  videos = list()


  for f in os.listdir(dirpath):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_extensions:
      continue
    videos.append(os.path.join(pre,f))

  print(videos)

  return videos

def start_algo(req):
  global n,pid
  n = os.fork()

  if n <= 0:
    pid = n
    os.system('python3 hello.py')
  
  return render_to_response('templates/protecc.html', [], request=req)

def stop_algo(req):
  global pid

  os.system('kill -2 ' + str(pid))

  return render_to_response('templates/protecc.html', [], request=req)



''' Route Configurations '''
if __name__ == '__main__':
  #app.run(host='0.0.0.0')
  config = Configurator()

  config.include('pyramid_jinja2')
  config.add_jinja2_renderer('.html')

  config.add_route('get_home', '/')
  config.add_view(get_home, route_name='get_home')

  config.add_route('unknown_dir', '/unknown_dir')
  config.add_view(unknown_dir, route_name='unknown_dir')

  config.add_route('motion_dir', '/motion_dir')
  config.add_view(motion_dir, route_name='motion_dir')

  config.add_route('trust_dir', '/trust_dir')
  config.add_view(trust_dir, route_name='trust_dir')

  config.add_route('record_dir', '/record_dir')
  config.add_view(record_dir, route_name='record_dir')

  config.add_route('start_algo', '/start_algo')
  config.add_view(start_algo, route_name='start_algo')

  config.add_route('stop_algo', '/stop_algo')
  config.add_view(stop_algo, route_name='stop_algo')

  config.add_static_view(name='/', path='./public', cache_max_age=3600)

  app = config.make_wsgi_app()
  server = make_server('0.0.0.0', 6000, app)
  server.serve_forever()