FROM python:3.8
  
RUN mkdir /datium_a

WORKDIR /datium_a

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /datium_a/

EXPOSE 5000

CMD "/datium_a/run_all.sh"