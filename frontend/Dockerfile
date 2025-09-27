FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Default to dev server (override with docker run or compose)
CMD ["npm", "run", "dev"]
