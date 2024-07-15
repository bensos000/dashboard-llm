from pandasai.llm.local_llm import LocalLLM
import streamlit as st
from pandasai.connectors import PostgreSQLConnector
from pandasai import SmartDataframe
import psycopg2
from sqlalchemy_schemadisplay import create_schema_graph
from sqlalchemy import MetaData
import base64
import cv2
import json

def getDataFrameFromTable(table, host, port, username, password, database):
    connector = PostgreSQLConnector(
        config={
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database,
            "table": table
        }
    )
    model = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3"
    )
    df = SmartDataframe(connector, config={
        "llm": model
    })
    return df

def render_img_html(image_b64):
    st.markdown(f"<img style='max-width: 100%;max-height: 100%;' src='data:image/png;base64, {image_b64}'/>", unsafe_allow_html=True)

def image_to_base64(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    _, encoded_image = cv2.imencode(".png", image)
    base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    return base64_image

def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

def init(filename, db_user, db_password, db_host, db_port, db_name):
    graph = create_schema_graph(metadata=MetaData(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'))
    graph.write_png(filename)
    conn = psycopg2.connect(f"dbname={db_name} user={db_user} password={db_password} host={db_host} port={db_port}")
    cur = conn.cursor()
    cur.execute("select * from information_schema.tables where table_schema = 'public'")
    alltables = [ i[2] for i in cur.fetchall() if i[3] == "BASE TABLE" ]
    return alltables


config = load_config()
db_config = config.get('database', {})

db_host = db_config.get('db_host', '')
db_name = db_config.get('db_name', '')
db_user = db_config.get('db_user', '')
db_password = db_config.get('db_password', '')
db_port = db_config.get('db_port', '')

filename = 'database.png'

st.set_page_config(page_title='Dashboard Analytics', layout = 'wide', page_icon = filename, initial_sidebar_state = 'auto')

st.title("Dashboard Analytics")

render_img_html(image_to_base64(filename))

alltables = init(filename,db_user, db_password, db_host, db_port, db_name)

table = st.selectbox("Select a table", alltables)

prompt = st.text_input("Enter your prompt")

if (st.button("Generate")):
    if prompt:
        with st.spinner("Generating..."):
            result = getDataFrameFromTable(table, db_host, db_port, db_user, db_password, db_name).chat(prompt)
            st.write(result)
            if isinstance(result, str) and result.find('.png') != -1:
                render_img_html(image_to_base64(result))