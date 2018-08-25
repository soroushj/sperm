#!/usr/bin/env python3

import os
import psycopg2 as pg

query = 'INSERT INTO results VALUES (' + ','.join(['%s'] * 27) + ')'
con = pg.connect(
    host='localhost',
    port=5432,
    dbname='sperm',
    user=os.environ['PGUSER'],
    password=os.environ['PGPASSWORD']
)
cur = con.cursor()

for path, _, files in os.walk('saved/results'):
    for name in files:
        run = path[-1]
        model = name[1]
        label = name[3]
        flags = name[5]
        with open(os.path.join(path, name)) as csv:
            next(csv)
            for line in csv:
                row = line.strip().split(',')
                values = [run, model, label, flags] + row
                cur.execute(query, values)

cur.close()
con.commit()
con.close()
