# 
FROM python:3.9

# 
WORKDIR /app

# 
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 
COPY . ./app

# 
CMD ["fastapi", "run", "app/main.py", "--port", "8080"]