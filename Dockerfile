FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY knn_euclidean.py .

CMD [ "knn_euclidean.lambda_controller" ]
