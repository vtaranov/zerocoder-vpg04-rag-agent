import unittest
from unittest.mock import patch
import httpx
from rag_agent import split_text, validate_url, cat_fact


class AgentTests(unittest.TestCase):
    def test_chunk_overlap_and_coverage(self):
        text = ''.join(str(n % 10) for n in range(2517))
        parts = split_text(text)
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(len(p) <= 1000 for p in parts))
        self.assertEqual(parts[0][-150:], parts[1][:150])
        self.assertEqual(parts[0] + ''.join(p[150:] for p in parts[1:]), text)
        self.assertEqual(split_text(''), [])
        with self.assertRaises(ValueError):
            split_text('x', 10, 10)

    def test_private_urls_rejected(self):
        for url in ['file:///etc/passwd', 'http://user:secret@example.com', 'http://127.0.0.1', 'http://[::1]', 'http://169.254.169.254']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url)

    def test_api_errors(self):
        request = httpx.Request('GET', 'https://catfact.ninja/fact')
        for response in [httpx.Response(503, request=request),
                         httpx.Response(200, request=request, json={'other': 'missing fact'}),
                         httpx.Response(200, request=request, text='not json')]:
            with patch('rag_agent.httpx.get', return_value=response):
                self.assertFalse(cat_fact()['ok'])
        with patch('rag_agent.httpx.get', side_effect=httpx.ReadTimeout('timeout')):
            self.assertEqual(cat_fact()['error'], 'ReadTimeout')


if __name__ == '__main__':
    unittest.main()
