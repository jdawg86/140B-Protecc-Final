FROM python:3.8
ENV PYTHONUNBUFFERED 1

# If using Bjoern webserver, please uncomment these two lines
RUN apt-get update
RUN apt-get install libev-dev -y
RUN pip install cmake

COPY ./src/requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN pip install typing_extensions
RUN pip install Flask
RUN pip install Image

WORKDIR /app

EXPOSE 6000

CMD python server.py
# Uncomment if you wish to use bjoern or waitress web servers instead of the default
# CMD python server_bjoern.py
# CMD python server_waitress.py