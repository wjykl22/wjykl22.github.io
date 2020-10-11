---
title: Kong和Keycloak第三方认证和权限管理
mathjax: true
abbrlink: fbb6b1b9
date: 2020-07-27 10:48:15
tags:
	- Kong
	- keycloak
	- 权限认证
	- docker
categories:
	- API网关
	- Kong&Konga
	- 权限认证
---

## 介绍

使用Keycloak作为第三方认证服务，Kong作为API网关与Keycloak进行对接，打通两者用户数据库，并使用Kong当中的ACL插件进行接口权限的设计。主要涉及到如下框架：

- Kong——开源API网关
- Keycloak——一个OpenID认证服务
- Konga——API网关后台管理可视化界面

![Screen-Shot-2018-11-15-at-17.29.03](Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-17.29.03-1595831384730.png)

下文将记录Kong、Konga和Keycloak三者的安装和对接过程，并介绍权限打通的设计思路。以下是主要步骤：

1. 创建Dockerfile，创建带有kong-oidc插件的镜像
2. 构建上述镜像
3. 创建`docker-compose.yml`文件，配置Kong，Konga和Keycloak的相关信息
4. 启动kong-db服务
5. 运行`migrations`
6. 启动Kong服务
7. 验证kong-oidc插件是否可用
8. 使用StreamSet打通Keycloak和Kong用户数据库，进行数据同步
9. 添加kong-oidc插件和ACL插件
10. 测试认证是否可行

## 安装过程

### 创建Dockerfile文件

第一，我们需要创建有关于Kong的镜像。我们还需要在这个镜像的基础上安装kong-oidc插件。我们可以采用以下两种方法进行:

1. 修改现有的、正在运行的容器并提交更改
2. 创建Dockerfile文件，并镜像构建

我们将采用第二种方法，以下是Dockerfile文件

```shell
$ mkdir -p docker/kong
$ touch docker/kong/Dockerfile
```

使用vim创建并打开文件，写入以下内容：

```dockerfile
FROM kong:1.4.2-centos

LABEL description="Centos 7 + Kong 1.4.2 + kong-oidc plugin"

RUN yum install -y git unzip && yum clean all

RUN luarocks install kong-oidc
```

以上代码将会将安装kong1.4.2版本，以及在此基础上安装kong-oidc插件，接下来构建该文件：

```shell
$ docker build -t kong:1.4.2-centos-oidc docker/kong/
```

如果遇到`Warning: The directory '/root/.cache/luarocks' or its parent directory is not owned by the current user`就忽略。

### 安装及配置Kong

接下来创建`docker-compose.yml`文件

```shell
$ touch docker-compose.yml
```

打开这个文件，并进行如下配置

```dockerfile
version: '3.4'

networks: 
  kong-net:

volumes:
  kong-datastore:

services:
  kong-db:
    image: postgres:9.6
    volumes:
      - kong-datastore:/var/lib/postgresql/data
    networks:
      - kong-net
    ports:
      - "15432:5432"
    environment:
      POSTGRES_DB:       api-gw
      POSTGRES_USER:     kong
      POSTGRES_PASSWORD: kong

  kong:
    image: kong:1.4.2-centos-oidc
    depends_on:
      - kong-db
    networks:
      - kong-net
    ports:
      - "8000:8000" # Listener
      - "8001:8001" # Admin API
      - "8443:8443" # Listener  (SSL)
      - "8444:8444" # Admin API (SSL)
    environment:
      KONG_DATABASE:         postgres
      KONG_PG_HOST:          kong-db
      KONG_PG_PORT:          5432
      KONG_PG_DATABASE:      api-gw
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
      KONG_PROXY_ERROR_LOG:  /dev/stderr
      KONG_ADMIN_ERROR_LOG:  /dev/stderr
      KONG_PROXY_LISTEN:     0.0.0.0:8000, 0.0.0.0:8443 ssl
      KONG_ADMIN_LISTEN:     0.0.0.0:8001, 0.0.0.0:8444 ssl
      KONG_PLUGINS:          bundled,oidc
```

接下来，采用下面的命令启动kong-db服务，其中`-d`是告诉docker在后台运行Docker Compose进程

```shell
$ docker-compose up -d kong-db
```

验证服务是否已经启动

```shell
$ docker-compose ps
```

接下来将迁移kong-db数据库，采用`migrations`命令。下面这个命令将驱动一个kong服务，`-rm`表示该服务将在命令运行之后关闭。

```shell
$ docker-compose run --rm kong kong migrations up
```

最后，我们将Kong启动，并且检查其运行状态

```shell
$ docker-compose up -d kong
$ docker-compose ps
```

检查Kong的管理API，检查OIDC插件是否可以在服务器上使用

```shell
$ curl -s http://localhost:8001 | jq .plugins.available_on_server.oidc
```

将返回`true`。但是，虽然OIDC插件在Kong上可用，但是还没有配置Keycloak，实际上还不能进行API的保护。

至此为止，就完成了Kong的安装

---

### 安装Konga可视化终端

Konga是开源的Kong可视化界面，方面API网关管理人员对Kong进行可视化管理，安装方法和Kong类似。

配置`docker-compose.yml`文件，添加如下内容：

```dockerfile
konga-prepare:
     image: pantsel/konga:next
     command: "-c prepare -a postgres -u postgresql://kong:kong@kong-db:5432/konga_db"
     networks:
       - kong-net
     restart: on-failure
     links:
       - kong-db
     depends_on:
       - kong-db
```

首先将konga的postgresql数据启动：

```shell
$ docker-compose up -d konga-prepare
```

检查是否启动成功：

```shell
$ docker-compose ps
```

接下来在`docker-compose.yml`再配置Konga镜像：

```dockerfile
 konga:
     image: pantsel/konga:latest
     networks:
       - kong-net
     environment:
       DB_ADAPTER: postgres
       DB_HOST: kong-db
       DB_USER: kong
       DB_DATABASE: konga_db
       NODE_ENV: production
       DB_PASSWORD: kong
     depends_on: 
       - kong-db
     ports:
       - "1337:1337"
```

再次启动konga

```shell
$ docker-compose up -d konga
```

检查是否启动成功：

```shell
$ docker-compose ps
```

用浏览器访问“IP地址:1337”端口，注册后台管理用户名密码，检查是否能够正常访问，进入后台管理界面后，配置和Kong的对接：

<img src="Kong和Keycloak第三方认证和权限管理/image-20200727141443389.png" alt="image-20200727141443389" style="zoom:67%;" />

对接完成后，可以用Konga查看并配置Kong网关的信息了。

### 安装Keycloak

本节欸将重点配置Keycloak的安装。我们还是采用docker对Keycloak进行安装。

![Screen-Shot-2018-11-15-at-19.38.47](Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-19.38.47.png)

在本节中，我们将运行以下步骤：

1. 修改`docker-compose.yml`来添加Keycloak和它的可视化终端服务
2. 注册Keycloak数据库服务
3. 注册Keycloak服务
4. 登录Keycloak
5. 为Kong添加Keycloak终端
6. 添加新用户

#### 使用Docker安装Keycloak

重新打开`docker-compose.yml`.为Keycloak添加网络：

```dockerfile
networks: 
  kong-net:
  keycloak-net:

volumes:
  kong-datastore:
  keycloak-datastore:
```

接下来将Keycloak数据库添加到`docker-compose.yml`中：

```dockerfile
services:
  ...
  keycloak-db:
    image: postgres:9.6
    volumes: 
      - keycloak-datastore:/var/lib/postresql/data
    networks:
      - keycloak-net
    ports:
      - "25432:5432"
    environment:
      POSTGRES_DB:       keycloak
      POSTGRES_USER:     keycloak
      POSTGRES_PASSWORD: password
```

启动服务

```shell
$ docker-compose up -d keycloak-db
```

验证是否可以使用（确保它的状态是启动的）

```shell
$ docker-compose ps
```

下一步，将Keycloak添加到`docker-compose.yml`的services中：

```dockerfile
services:
  ...
  keycloak:
    image: jboss/keycloak:4.5.0.Final
    depends_on:
      - keycloak-db
    networks:
      - keycloak-net
    ports:
      - "8180:8080"
    environment:
      DB_VENDOR:   POSTGRES
      DB_ADDR:     keycloak-db
      DB_PORT:     5432
      DB_DATABASE: keycloak
      DB_USER:     keycloak
      DB_PASSWORD: password
      KEYCLOAK_USER:     admin
      KEYCLOAK_PASSWORD: admin
```

最后启动Keycloak服务：

```shell
$ docker-compose up -d keycloak
```

验证服务是否可用

```shell
$ docker-compose ps
```

#### 添加客户端到Keycloak中

点击Administration Console进入到控制台当中，此时需要输入用户名和密码：

在`Master`域下，创建客户端：

![Screen-Shot-2018-11-15-at-20.38.29](Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-20.38.29.png)

在添加客户端页面，填写“客户端ID”为kong，点击保存按钮。

![Screen-Shot-2018-11-15-at-20.43.23](Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-20.43.23.png)

在详情页面中，可以看到“Access Type”，我们需要选择的是“Confidential”，其中“Root URL”是“IP地址:8000”（Kong所接管的端口），以及“Valid Redirect URIs”填写“/*”就可以了，我们的设置如下：



![image-20200727135258550](Kong和Keycloak第三方认证和权限管理/image-20200727135258550.png)

点击保存之后，在“Credentials”页面中会有对应的“Secret”，这个密钥是需要在Kong当中进行配置的

![Screen-Shot-2018-11-15-at-20.52.51](Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-20.52.51.png)

#### 为Keycloak添加用户

要添加用户，单击左侧侧边栏的“Users”选项卡，然后单击右侧的“add user”按钮

![Screen-Shot-2018-11-15-at-20.58.07](Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-20.58.07.png)

在下一页，将“Username”设置为“用户”，并将“Email Enable”开关设置为“On”。然后，点击“Save”按钮。

<img src="Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-21.00.07.png" alt="Screen-Shot-2018-11-15-at-21.00.07" style="zoom:67%;" />

点击“Credentials”选项卡，输入密码，确认，确保“Temporary”开关设置为“关闭”。然后，点击“Save”按钮。

至此，在Keycloak当中的配置已经完成了。

---

### 将Kong和Keycloak进行对接

接下来将配置Keycloak和Kong的对接部分

<img src="Kong和Keycloak第三方认证和权限管理/Screen-Shot-2018-11-15-at-21.06.44.png" alt="Screen-Shot-2018-11-15-at-21.06.44" style="zoom:67%;" />

1. 创建服务和路由（此处省略）

2. 在全局范围内安装oidc插件

   点击左侧“Plugin”后，再点击“ADD GLOBAL PLUGINS”，在Other当中，又OIDC插件，找到并点击“ADD PLUGIN”，会弹出如下表单，表单当中的条目信息可以参考：https://github.com/nokia/kong-oidc在GitHub上的首页。

   其中以下几个需要重点关注：

   - client_id：需要和Keycloak中创建的client_id对应一致
   - client_secret：需要需要和Keycloak中自动生成的secret一致
   - realm：默认就是Keycloak中的master域
   - redirect after logout uri：/
   - discovery：根据OIDC填写：http://XXXXXX:8180/auth/realms/master/.well-known/openid-configuration

3. 配置完成后，访问Kong接管的API接口，会自动跳转到Keycloak的登录界面，说明Keycloak和Kong对接完成

   <img src="Kong和Keycloak第三方认证和权限管理/image-20200727142400435.png" alt="image-20200727142400435" style="zoom:67%;" />

## 存在问题

官方给出了以上配置虽然打通了Kong和Keycloak，但是由于登录是Keycloak进行管理，采用的是Keycloak中的用户，而这些用户和Kong当中的“Consumer”是分离的，而且此时不能够通过Keycloak进行用户权限的管理。

如果将用户管理交给Keycloak，就会架空Kong当中的Consumer，最好是有办法将Kong中的Consumer和Keycloak中的User打通

### 目标

- 打通Keycloak和Kong中的用户表
- 能够使得ACL插件进行白名单和黑名单的访问控制

### 方法

1. 使用ETL工具两者数据表，将ID和用户名进行同步
2. 修改ACL插件使其能够获取到认证过后的`authenticated_consumer `

### 操作步骤

#### ETL工具同步

此处过程省略

#### 修改ACL插件

在kong-oidc插件官方文档中，给出了`X-Userinfo`是被注入在请求头当中的，例如：

```json
X-Userinfo: {"preferred_username":"alice","id":"60f65308-3510-40ca-83f0-e9c0151cc680","sub":"60f65308-3510-40ca-83f0-e9c0151cc680"}
```

而该插件同样在`ngx.ctx.authenticated_consumer`中设置了变量，他能够支持其他插件对认证通过用户进行操作，因此，可以让ACL获取该信息，利用该信息和现有“Consumer”做匹配。

```json
ngx.ctx.authenticated_credential = {
    id = "60f65308-3510-40ca-83f0-e9c0151cc680",   -- sub field from Userinfo
    username = "alice"                             -- preferred_username from Userinfo
}
```

修改如下：

- 进入docker容器当中

  ```shell
  $ sudo docker exec -it  容器ID /bin/bash
  ```

- 进入Kong插件所在目录

  ```shell
  $ cd /usr/local/share/lua/5.1/kong/plugins/acl
  ```

- 打开group.lua文件，对`get_current_consumer_id`做修改

  ```diff
  local function get_current_consumer_id()
    kong.log.err("oidc info:", ngx.ctx.authenticated_credential.id)
    return (kong.client.get_consumer() or EMPTY).id or
           (kong.client.get_credential() or EMPTY).consumer_id or
  +        (ngx.ctx.authenticated_credential or EMPTY).id
  ```

这样就可以做到Kong和Keycloak真正打通。
