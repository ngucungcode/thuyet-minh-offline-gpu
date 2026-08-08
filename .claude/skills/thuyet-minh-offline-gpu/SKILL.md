```markdown
# thuyet-minh-offline-gpu Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the `thuyet-minh-offline-gpu` TypeScript codebase. You'll learn about file organization, import/export styles, commit message habits, and how to write and run tests. While no specific framework or automated workflows are detected, this guide will help you maintain consistency and efficiency when contributing to the project.

## Coding Conventions

### File Naming
- Use **camelCase** for file names.
  - Example: `audioProcessor.ts`, `textToSpeechService.ts`

### Import Style
- Use **relative imports** for internal modules.
  - Example:
    ```typescript
    import { processAudio } from './audioProcessor';
    ```

### Export Style
- Use **named exports** rather than default exports.
  - Example:
    ```typescript
    // audioProcessor.ts
    export function processAudio(input: Buffer): Buffer { ... }
    ```

### Commit Messages
- Freeform style, usually short (average 28 characters).
- No strict prefixing required.

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or module.
**Command:** `/add-feature`

1. Create a new file using camelCase naming.
2. Implement your feature using TypeScript.
3. Use relative imports to include any dependencies.
4. Export your functions or classes using named exports.
5. Write a corresponding test file named `yourFeature.test.ts`.
6. Commit your changes with a concise message.

### Fixing a Bug
**Trigger:** When correcting an error or issue in the codebase.
**Command:** `/fix-bug`

1. Locate the problematic code.
2. Apply the necessary fix.
3. Update or add tests in the relevant `*.test.ts` file.
4. Commit your fix with a clear, short message.

### Writing and Running Tests
**Trigger:** When verifying code functionality.
**Command:** `/run-tests`

1. Write test files following the `*.test.ts` pattern.
2. Use your preferred testing framework (not specified in repo).
3. Run tests using the framework's CLI (e.g., `npx jest` or `npx mocha`).
4. Ensure all tests pass before committing.

## Testing Patterns

- Test files are named using the pattern `*.test.ts`.
- The specific testing framework is not enforced; choose one that fits your needs (e.g., Jest, Mocha).
- Place test files alongside the modules they test or in a dedicated `tests` directory.
- Example test file:
  ```typescript
  // audioProcessor.test.ts
  import { processAudio } from './audioProcessor';

  describe('processAudio', () => {
    it('should process audio buffer correctly', () => {
      // test implementation
    });
  });
  ```

## Commands
| Command      | Purpose                                     |
|--------------|---------------------------------------------|
| /add-feature | Scaffold and implement a new feature/module  |
| /fix-bug     | Apply and commit a bug fix                  |
| /run-tests   | Run all test suites in the codebase         |
```