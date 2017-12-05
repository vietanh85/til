import time
import pika
import json
import requests

import psycopg2.pool

from flask import Flask, render_template, redirect, flash, request
app = Flask(__name__)
app.secret_key = 'IM_SO_SECRET'

# number of records need to be inserted
N = 10000
# inserted 10000 records within 318.8598828315735 seconds (finished)

def connect_q():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    return connection.channel()

channel = connect_q()
channel.queue_declare(queue='run')
pool = psycopg2.pool.ThreadedConnectionPool(1, 10, user='postgres', host='localhost', database='postgres')

def execute(sql, commit=True):
    try:
        conn = pool.getconn()
    except psycopg2.Error as e:
        raise e
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            if commit:
                conn.commit()
        except psycopg2.Error as e:
            raise e
        finally:
            pool.putconn(conn)

def query(sql):
    rows = []
    try:
        conn = pool.getconn()
    except psycopg2.Error as e:
        raise e
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            rows = cur.fetchall()
        except psycopg2.Error as e:
            raise e
        finally:
            pool.putconn(conn)
    return rows

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add-file')
def add_file(f=True):
    start = time.time()
    # sleep for 30 miliseconds
    time.sleep(0.03)
    name = '{}.tsv'.format(time.time())
    execute('INSERT INTO files(name) VALUES (\'{}\')'.format(name))
    end = time.time()
    if f:
        flash('inserted 1 record within {} seconds (finished)'.format((end - start)))
    return redirect('/results')

@app.route('/old-run')
def old_run():
    start = time.time()
    results = []
    for i in range(N):
        add_file(False)
    end = time.time()
    flash('inserted {} records within {} seconds (finished)'.format(N, (end - start)))
    return redirect('/results')


@app.route('/run')
def run():
    # publish to rabbit mq right here
    start = time.time()
    for i in range(N):
        params = '{}.tsv'.format(time.time())
        print('sending {}'.format(params))
        try:
            channel.basic_publish(exchange='', routing_key='run', body=params)
        except:
            # re-connect
            channel = connect_q()
            channel.basic_publish(exchange='', routing_key='run', body=params)
    end = time.time()
    flash('sent {} messages within {} seconds (still processing the messages)'.format(N, (end - start)))
    return redirect('/results?refresh=yes')

@app.route('/results')
def results():
    refresh = True if request.args.get('refresh') is not None else False
    # return refresh
    res = query('SELECT name FROM files ORDER BY id DESC LIMIT 100')
    count = query('SELECT count(id) FROM files')
    return render_template('result.html', results=[i[0] for i in res], count=count[0][0], refresh=refresh)

@app.route('/clear')
def clear():
    execute('DELETE FROM files')
    return redirect('/')

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)