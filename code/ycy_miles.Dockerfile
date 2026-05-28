FROM radixark/miles:latest

RUN apt update \
    && apt install -y pciutils  iproute2 pdsh vim wget htop git  language-pack-zh-hans openssh-server sudo


RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
mkdir -p /var/run/sshd

RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN for i in `locale | awk -F'=' '{print $1}'` ; do export  $i="zh_CN.UTF-8"; done

RUN groupadd -f -g 2216 nlp-intern && groupadd -f -g 2239 nlp-train && groupadd -f -g 3001 docker && groupadd -f -g 2243 yangchengyi && useradd -m -u 2243 -g yangchengyi -G nlp-intern,nlp-train,docker -s /bin/bash yangchengyi \
    && echo 'yangchengyi ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && echo "yangchengyi:Yangcy173950." | chpasswd 

# COPY init-home-symlinks.sh /usr/local/bin/init-home-symlinks.sh
# RUN chmod +x /usr/local/bin/init-home-symlinks.sh
# RUN chown yangchengyi:nlp-train /home/yangchengyi -R
USER yangchengyi
WORKDIR /fs/nlp-intern/yangchengyi