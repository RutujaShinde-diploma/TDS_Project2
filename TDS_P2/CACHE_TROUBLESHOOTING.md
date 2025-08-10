# Cache Troubleshooting Guide

## 🚨 Problem: Getting Old Results After File Updates

If you're getting results from an old version of your test file (e.g., `wiki_test.txt`), this is a caching issue. The system caches execution plans and LLM responses to improve performance, but sometimes this can cause problems during development and testing.

## 🔍 Why This Happens

The system uses multiple layers of caching:

1. **Plan Caching**: Execution plans are cached based on content hash
2. **LLM Response Caching**: Code generation responses are cached
3. **Code Pattern Caching**: Working code patterns are cached

When you update a file, the content hash changes, but there might be residual caching at other levels.

## 🛠️ Solutions

### 1. **Clear All Caches (Recommended for Testing)**

```bash
# Clear all caches
curl -X POST "http://localhost:8000/api/cache/clear"
```

### 2. **Use Cache Bypass Parameter**

When submitting a job, add the `bypass_cache` parameter:

```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions=@wiki_test.txt" \
  -F "bypass_cache=true"
```

### 3. **Clear Specific Plan Cache**

Clear cache for specific content:

```bash
curl -X POST "http://localhost:8000/api/cache/clear/plan" \
  -d "questions=$(cat wiki_test.txt)"
```

### 4. **Check Cache Status**

Monitor what's cached:

```bash
# Basic status
curl "http://localhost:8000/api/cache/status"

# Detailed debug info
curl "http://localhost:8000/api/cache/debug"
```

## 🧪 Testing Cache Management

Use the provided test script:

```bash
python test_cache_clear.py
```

This script will:
- Check cache status
- Clear all caches
- Test cache bypass functionality
- Verify cache clearing worked

## 🔧 Development Workflow

### During Development:
1. **Always use `bypass_cache=true`** when testing changes
2. **Clear caches frequently** with the clear endpoint
3. **Monitor cache status** to see what's cached

### Before Production:
1. **Remove `bypass_cache`** for normal operation
2. **Keep caching enabled** for performance
3. **Monitor cache hit rates** for optimization

## 📊 Cache Debugging

### Check What's Cached:
```bash
curl "http://localhost:8000/api/cache/debug"
```

This will show:
- Redis connection status
- Number of cached items
- Specific cache keys for plans, LLM responses, and code
- Memory usage and performance metrics

### Common Cache Patterns:
- `plan:*` - Execution plans
- `llm:*` - LLM responses
- `code:*` - Code patterns

## 🚀 Performance vs. Freshness Trade-off

| Scenario | Cache Setting | Performance | Freshness |
|----------|---------------|-------------|-----------|
| **Development** | `bypass_cache=true` | Lower | ✅ Always Fresh |
| **Testing** | Clear caches manually | Lower | ✅ Fresh after clearing |
| **Production** | Default (cached) | ✅ High | ⚠️ Cached until TTL |

## 🔄 Automatic Cache Invalidation

The system now includes **file modification timestamps** in cache keys, so:
- ✅ **File content changes** automatically invalidate cache
- ✅ **File modification time changes** automatically invalidate cache
- ✅ **New files** get fresh execution

## 📝 Best Practices

1. **During Development**: Use `bypass_cache=true`
2. **Before Testing**: Clear caches with `/api/cache/clear`
3. **Monitor Cache**: Use `/api/cache/debug` to see what's cached
4. **File Changes**: Update file modification times (touch file) if needed
5. **Production**: Keep caching enabled for performance

## 🆘 If Problems Persist

1. **Check file timestamps**: Ensure files have recent modification times
2. **Verify cache clearing**: Use debug endpoints to confirm caches are empty
3. **Restart service**: Sometimes a service restart helps
4. **Check logs**: Look for cache-related error messages

## 📚 Related Endpoints

- `POST /api/cache/clear` - Clear all caches
- `GET /api/cache/status` - Basic cache status
- `GET /api/cache/debug` - Detailed cache information
- `POST /api/cache/clear/plan` - Clear specific plan cache
- `POST /api/` with `bypass_cache=true` - Force fresh execution

---

**Remember**: Caching is designed to improve performance, but during development and testing, you often want fresh results. Use the cache management tools to control when caching is helpful vs. when it's getting in the way.
