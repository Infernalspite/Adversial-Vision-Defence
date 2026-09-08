import { render, screen } from '@testing-library/react'
import App from './App'

test('renders the ARGUS-AEGIS shell', () => {
  render(<App />)
  expect(screen.getByText('ARGUS-AEGIS')).toBeInTheDocument()
  expect(screen.getByText('Mission control for uncertain vision.')).toBeInTheDocument()
  expect(screen.getByText('Attack Lab')).toBeInTheDocument()
})
