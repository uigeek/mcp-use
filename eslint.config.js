import antfu from '@antfu/eslint-config'

export default antfu({
  formatters: true,
  typescript: true,
  rules: {
    'node/prefer-global/process': 'off',
  },
}, {
  files: ['examples/**/*.ts'],
  rules: {
    'no-console': 'off',
  },
})
