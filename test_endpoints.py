import unittest
import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ['FLASK_SECRET_KEY'] = 'test-secret-key-for-testing'
os.environ['DB_HOST'] = os.getenv('DB_HOST', 'localhost')
os.environ['DB_USER'] = os.getenv('DB_USER', 'root')
os.environ['DB_PASSWORD'] = os.getenv('DB_PASSWORD', '')
os.environ['DB_NAME'] = 'test_ai_tutor_db'

from app import app
import pymysql


class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()
        self.test_db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': 'test_ai_tutor_db'
        }
        self._init_test_db()

    def _init_test_db(self):
        try:
            conn = pymysql.connect(**self.test_db_config)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100),
                    email VARCHAR(100) UNIQUE,
                    password VARCHAR(255)
                )
            ''')
            conn.commit()
            conn.close()
        except Exception:
            pass

    def tearDown(self):
        try:
            conn = pymysql.connect(**self.test_db_config)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE email LIKE %s', ('test_%',))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def test_home_route_renders_index_for_guests(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Transparent learning workflow', response.data)

    def test_home_route_shows_index_for_logged_in_users(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'Test User'
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Try the tutor', response.data)

    def test_register_get(self):
        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Create Account', response.data)

    def test_register_post_success(self):
        response = self.client.post('/register', data={
            'name': 'Test User',
            'email': 'test_register@example.com',
            'password': 'testpass123'
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.location)

    def test_register_post_duplicate_email(self):
        self.client.post('/register', data={
            'name': 'Test User 1',
            'email': 'test_duplicate@example.com',
            'password': 'testpass123'
        })
        response = self.client.post('/register', data={
            'name': 'Test User 2',
            'email': 'test_duplicate@example.com',
            'password': 'testpass456'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'already exists', response.data)

    def test_register_post_missing_fields(self):
        response = self.client.post('/register', data={
            'name': 'Test User',
            'email': '',
            'password': 'testpass123'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'All fields are required', response.data)

    def test_login_get(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Secure Login', response.data)

    def test_login_post_success(self):
        self.client.post('/register', data={
            'name': 'Test Login User',
            'email': 'test_login@example.com',
            'password': 'testpass123'
        })
        response = self.client.post('/login', data={
            'email': 'test_login@example.com',
            'password': 'testpass123'
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/chat', response.location)

    def test_login_post_invalid_credentials(self):
        response = self.client.post('/login', data={
            'email': 'nonexistent@example.com',
            'password': 'wrongpass'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid credentials', response.data)

    def test_login_post_wrong_password(self):
        self.client.post('/register', data={
            'name': 'Test User',
            'email': 'test_wrongpass@example.com',
            'password': 'correctpass'
        })
        response = self.client.post('/login', data={
            'email': 'test_wrongpass@example.com',
            'password': 'wrongpass'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid credentials', response.data)

    def test_logout(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'Test User'
        response = self.client.get('/logout', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.location)

    def test_chat_requires_login(self):
        response = self.client.get('/chat', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.location)

    def test_chat_with_session(self):
        with self.client.session_transaction() as sess:
            sess['user'] = 'Test User'
        response = self.client.get('/chat')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test User', response.data)

    def test_ask_endpoint_requires_message(self):
        response = self.client.post('/ask', data={})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('response', data)
        self.assertIn('source', data)
        self.assertEqual(data['source'], 'uncertain')

    def test_ask_endpoint_with_message(self):
        response = self.client.post('/ask', data={'message': 'What is Python?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('response', data)
        self.assertIn('source', data)
        self.assertIn(data['source'], ['dataset', 'llm', 'uncertain'])

    def test_ask_endpoint_empty_message(self):
        response = self.client.post('/ask', data={'message': ''})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('response', data)

    def test_status_endpoint(self):
        response = self.client.get('/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn('mode', data)
        self.assertEqual(data['status'], 'Chatbot is running')

    def test_full_user_flow(self):
        response = self.client.post('/register', data={
            'name': 'Flow Test User',
            'email': 'test_flow@example.com',
            'password': 'flowpass123'
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)

        response = self.client.post('/login', data={
            'email': 'test_flow@example.com',
            'password': 'flowpass123'
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)

        response = self.client.get('/chat')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Flow Test User', response.data)

        response = self.client.post('/ask', data={'message': 'What is a list?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('response', data)

        response = self.client.get('/logout', follow_redirects=False)
        self.assertEqual(response.status_code, 302)

        response = self.client.get('/chat', follow_redirects=False)
        self.assertEqual(response.status_code, 302)


if __name__ == '__main__':
    unittest.main()

