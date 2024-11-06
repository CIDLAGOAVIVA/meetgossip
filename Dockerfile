# Use uma imagem base do Python
FROM python:3.10-slim

# Instale ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos de requisitos
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação
COPY . .

# Exponha a porta que o Streamlit usa
EXPOSE 8500

# Comando para rodar a aplicação
CMD ["streamlit", "run", "meetgossip.py"]