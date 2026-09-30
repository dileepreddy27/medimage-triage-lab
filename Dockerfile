FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY medimage ./medimage
RUN pip install --no-cache-dir . && useradd --create-home researcher && mkdir /results && chown researcher /results
USER researcher
ENTRYPOINT ["medimage"]
CMD ["demo", "--out", "/results/demo"]
