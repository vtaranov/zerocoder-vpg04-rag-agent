"""Реальная проверка API; результаты пишутся по этапам, включая ошибки."""
import json
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
from rag_agent import RAGAgent, cat_fact

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence'
URL = 'https://www.python.org/doc/essays/blurb/'


def main():
    OUT.mkdir(exist_ok=True)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'checks': [],
              'versions': {p: importlib.metadata.version(p) for p in
                           ['langchain', 'langchain-openai', 'openai', 'pinecone', 'beautifulsoup4']}}

    def check(name, action, validate):
        try:
            result = action()
            passed = bool(validate(result))
            entry = {'name': name, 'passed': passed, 'result': result}
        except Exception as exc:
            entry = {'name': name, 'passed': False, 'error_type': type(exc).__name__}
        report['checks'].append(entry)
        (OUT / 'live-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(name, 'PASS' if entry['passed'] else 'FAIL', flush=True)
        return entry

    agent = RAGAgent()
    report['index'] = agent.index_name
    report['namespace'] = agent.namespace
    report['model'] = agent.chat_model
    before = agent.index.describe_index_stats().to_dict()
    report['other_namespaces_before'] = {k: v for k, v in before.get('namespaces', {}).items() if k != agent.namespace}
    check('pinecone_probe', agent.probe, lambda r: r['dimension'] == 1536)
    ingestion = check('url_ingestion', lambda: agent.add_url(URL), lambda r: r['chunks'] > 0)
    check('url_search', lambda: agent.search('What is Python? Is Python portable?'),
          lambda r: bool(r) and all(m.get('source') == URL for m in r))
    check('rag_answer', lambda: agent.answer('По загруженной базе знаний: какие особенности языка Python перечислены в статье What is Python? Укажи источник.'),
          lambda r: any(t['tool'] == 'search_knowledge' for t in r['tools']) and URL in r['answer'])
    check('missing_answer', lambda: agent.answer('Используя только загруженную базу, сообщи точный размер моей арендной платы за ноябрь 2026. Если сведений нет, прямо скажи об этом.'),
          lambda r: any(t['tool'] == 'search_knowledge' for t in r['tools']))
    check('cat_api_direct', cat_fact, lambda r: r['ok'] and r['http_status'] == 200)
    check('cat_api_agent', lambda: agent.answer('Получи новый случайный факт о кошках через внешний API. Переведи его на русский и укажи источник.'),
          lambda r: any(t['tool'] == 'get_cat_fact' and json.loads(t['result']).get('ok') for t in r['tools']))
    check('agent_url_tool', lambda: agent.answer('Добавь в базу знаний эту страницу: ' + URL + ' . Сообщи число сохранённых чанков.'),
          lambda r: any(t['tool'] == 'add_url' and json.loads(t['result']).get('ok') for t in r['tools']))
    if ingestion['passed']:
        ids = ingestion['result']['ids']
        check('reingestion_same_ids', lambda: {'ids': sorted(agent.index.fetch(ids=ids, namespace=agent.namespace).vectors)},
              lambda r: r['ids'] == sorted(ids))
    after = agent.index.describe_index_stats().to_dict()
    report['other_namespaces_after'] = {k: v for k, v in after.get('namespaces', {}).items() if k != agent.namespace}
    report['other_namespace_counts_unchanged'] = report['other_namespaces_before'] == report['other_namespaces_after']
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['all_checks_passed'] = all(c['passed'] for c in report['checks']) and report['other_namespace_counts_unchanged']
    (OUT / 'live-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('all_checks_passed', report['all_checks_passed'], flush=True)
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
