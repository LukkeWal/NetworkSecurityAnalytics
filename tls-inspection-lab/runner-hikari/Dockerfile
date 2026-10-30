# Client container for the HIKARI-style run: the authors' scripts, the Selenium they were written for,
# Firefox and Chromium with their drivers, and a local mitmproxy on port 8080 (their scripts point browsers at it).
FROM python:3.9-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium chromium-driver firefox-esr git curl ca-certificates tar bzip2 build-essential libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN arch=$(uname -m) \
    && if [ "$arch" = aarch64 ]; then file=linux-aarch64; else file=linux64; fi \
    && curl -sSL "https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-$file.tar.gz" | tar xz -C /usr/local/bin

RUN pip install --no-cache-dir selenium==3.141.0 "urllib3<2" python-wordpress-xmlrpc psutil pandas requests mitmproxy==7.0.4

# mitmproxy 7 needs the Flask 1 era libraries
RUN pip install --no-cache-dir "Flask<2.1" "Werkzeug<2.1" "jinja2<3.1" "markupsafe<2.1" "itsdangerous<2.1"

RUN git clone --depth 1 https://github.com/andreysfc/generating-encrypted-network /lab \
    && mkdir -p /lab/aux/chrome /lab/aux/firefox /lab/profiles/aux/chrome /lab/profiles/aux/firefox \
    && for base in /lab/aux /lab/profiles/aux; do \
         ln -sf /usr/bin/chromedriver $base/chrome/chromedriver; ln -sf /usr/local/bin/geckodriver $base/firefox/geckodriver; \
       done

# LibDGA looks for its word lists under raw_data/
RUN mkdir -p /lab/raw_data && ln -s /lab/profiles/names /lab/raw_data/names && ln -s /lab/profiles/emailproviders /lab/raw_data/emailproviders

# Chromium refuses to run as root without these; the lab's certificate is self-signed
RUN printf 'export CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox --disable-dev-shm-usage --disable-gpu --ignore-certificate-errors"\n' > /etc/chromium.d/lab

COPY start.sh /start.sh
RUN chmod +x /start.sh
WORKDIR /lab
ENTRYPOINT ["/start.sh"]
