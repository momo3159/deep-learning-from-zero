FROM python:3

ARG FILEDIR

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD python $FILEDIR
