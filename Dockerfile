FROM python:3.8

WORKDIR /usr/d2m

RUN mkdir -p assets/data/raw

COPY src ./src
COPY dvc.yaml ./dvc.yaml
COPY params_default.yaml ./params.yaml
COPY params_default.yaml ./params_default.yaml
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install dvc
RUN pip3 install flask flask-restful

EXPOSE 5000

RUN dvc init --no-scm

CMD ["python3", "src/api.py"]
