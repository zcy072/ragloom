# 基础镜像前缀与 npm 源均可在构建时覆盖，见 docker-compose.yml。
ARG DOCKER_LIBRARY_PREFIX=docker.m.daocloud.io/library

# Step 1: Build Vite app
FROM ${DOCKER_LIBRARY_PREFIX}/node:20.20.2-alpine AS build
ARG NPM_REGISTRY=https://registry.npmmirror.com/
WORKDIR /app
COPY package*.json ./
# 使用锁文件保证本地、测试与镜像中的依赖版本一致。
RUN npm config set registry "${NPM_REGISTRY}" && npm ci
COPY . .
RUN npm run build

# Step 2: Serve with Nginx
FROM ${DOCKER_LIBRARY_PREFIX}/nginx:1.31.6-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
