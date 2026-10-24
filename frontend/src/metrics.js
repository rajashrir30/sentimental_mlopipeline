// Latest recorded runs from MLflow experiment 619827501377454495.
export const models = [
  { key: 'logistic', name: 'Logistic Regression', family: 'Baseline', run: '8f5c081b050a42e4a7da5892af8b94db', epochs: '—', test: { accuracy: .8888888888888888, precision: .877906976744186, recall: .9041916167664671, f1: .8908554572271387 }, validation: { accuracy: .8848848848848849, precision: .8829365079365079, recall: .8882235528942116, f1: .8855721393034826 } },
  { key: 'lstm', name: 'LSTM', family: 'Deep learning', run: '924cdd94c9c545c68b1fa1ec2c05b648', epochs: 2, test: { accuracy: .8228228228228228, precision: .8, recall: .8622754491017964, f1: .829971181556196 }, validation: { accuracy: .8158158158158159, precision: .7897623400365631, recall: .8622754491017964, f1: .8244274809160306 } }
]

export const experiments = [
  { model: 'LSTM', run: '924cdd94', started: '2026-10-19', accuracy: .8228228228228228, f1: .829971181556196, epochs: 2, status: 'Completed' },
  { model: 'Logistic Regression', run: '8f5c081b', started: '2026-10-19', accuracy: .8888888888888888, f1: .8908554572271387, epochs: '—', status: 'Completed' },
  { model: 'LSTM', run: 'e52ee9b7', started: '2026-10-18', accuracy: .8398398398398398, f1: .84251968503937, epochs: 2, status: 'Completed' },
  { model: 'Logistic Regression', run: '1290fa65', started: '2026-10-18', accuracy: .8888888888888888, f1: .8908554572271387, epochs: '—', status: 'Completed' }
]
