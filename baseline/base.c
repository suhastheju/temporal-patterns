/*
 * `Pattern detection in large temporal graphs using algebraic fingerprints`
 *
 * This experimental source code is supplied to accompany the
 * aforementioned paper.
 *
 * The source code is configured for a gcc build to a native
 * microarchitecture that must support the AVX2 and PCLMULQDQ
 * instruction set extensions. Other builds are possible but
 * require manual configuration of 'Makefile' and 'builds.h'.
 *
 * The source code is subject to the following license.
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 S. Thejaswi, A. Gionis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<sys/utsname.h>
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<ctype.h>
#include<omp.h>

/************************************************************* Configuration. */

#define MAX_K          32
#define MAX_SHADES     32

#define PREFETCH_PAD   32
#define MAX_THREADS   128

#define UNDEFINED -1
#define MATH_INF ((index_t)0x3FFFFFFF)

typedef long int index_t; // default to 64-bit indexing
typedef unsigned int shade_map_t;

#include"ffprng.h"   // fast-forward pseudorandom number generator

typedef unsigned long scalar_t;

/********************************************************************* Flags. */

index_t flag_bin_input    = 0; // default to ASCII input

/************************************************************* Common macros. */

/* Linked list navigation macros. */

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }


/*********************************************************** Error reporting. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d]\n"
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

/********************************************************* Get the host name. */

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

/********************************************************* Available threads. */

index_t num_threads(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/********************************************** Memory allocation & tracking. */

#define MALLOC(x) malloc_wrapper(x)
#define FREE(x) free_wrapper(x)

index_t malloc_balance = 0;

struct malloc_track_struct
{
    void *p;
    size_t size;
    struct malloc_track_struct *prev;
    struct malloc_track_struct *next;
};

typedef struct malloc_track_struct malloc_track_t;

malloc_track_t malloc_track_root;
size_t malloc_total = 0;

#define MEMTRACK_STACK_CAPACITY 256
size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

void *malloc_wrapper(size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = size;
    pnlinkprev(&malloc_track_root, t);
    malloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;    
    return p;
}

void free_wrapper(void *p)
{
    malloc_track_t *t = malloc_track_root.next;
    for(;
        t != &malloc_track_root;
        t = t->next) {
        if(t->p == p)
            break;
    }
    if(t == &malloc_track_root)
        ERROR("FREE issued on a non-tracked pointer %p", p);
    malloc_total -= t->size;
    pnunlink(t);
    free(t);
    
    free(p);
    malloc_balance--;
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC(sizeof(index_t)*n);
    return t;
}

void push_memtrack(void) 
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = malloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];    
}

size_t current_mem(void)
{
    return malloc_total;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

void print_current_mem(void)
{
    fprintf(stdout, "{curr: %.2lfGiB}", inGiB(current_mem()));
    fflush(stdout);
}

void print_pop_memtrack(void)
{
    fprintf(stdout, "{peak: %.2lfGiB}", inGiB(pop_memtrack()));
    fflush(stdout);
}

/******************************************************** Timing subroutines. */

#define TIME_STACK_CAPACITY 256
double start_stack[TIME_STACK_CAPACITY];
index_t start_stack_top = -1;

void push_time(void) 
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
    start_stack[++start_stack_top] = omp_get_wtime();
}

double pop_time(void)
{
    double wstop = omp_get_wtime();
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}

/******************************************************************* Sorting. */

void shellsort(index_t n, index_t *a)
{
    index_t h = 1;
    index_t i;
    for(i = n/3; h < i; h = 3*h+1)
        ;
    do {
        for(i = h; i < n; i++) {
            index_t v = a[i];
            index_t j = i;
            do {
                index_t t = a[j-h];
                if(t <= v)
                    break;
                a[j] = t;
                j -= h;
            } while(j >= h);
            a[j] = v;
        }
        h /= 3;
    } while(h > 0);
}

#define LEFT(x)      (x<<1)
#define RIGHT(x)     ((x<<1)+1)
#define PARENT(x)    (x>>1)

void heapsort_indext(index_t n, index_t *a)
{
    /* Shift index origin from 0 to 1 for convenience. */
    a--; 
    /* Build heap */
    for(index_t i = 2; i <= n; i++) {
        index_t x = i;
        while(x > 1) {
            index_t y = PARENT(x);
            if(a[x] <= a[y]) {
                /* heap property ok */
                break;              
            }
            /* Exchange a[x] and a[y] to enforce heap property */
            index_t t = a[x];
            a[x] = a[y];
            a[y] = t;
            x = y;
        }
    }

    /* Repeat delete max and insert */
    for(index_t i = n; i > 1; i--) {
        index_t t = a[i];
        /* Delete max */
        a[i] = a[1];
        /* Insert t */
        index_t x = 1;
        index_t y, z;
        while((y = LEFT(x)) < i) {
            z = RIGHT(x);
            if(z < i && a[y] < a[z]) {
                index_t s = z;
                z = y;
                y = s;
            }
            /* Invariant: a[y] >= a[z] */
            if(t >= a[y]) {
                /* ok to insert here without violating heap property */
                break;
            }
            /* Move a[y] up the heap */
            a[x] = a[y];
            x = y;
        }
        /* Insert here */
        a[x] = t; 
    }
}

/******************************************************* Bitmap manipulation. */

void bitset(index_t *map, index_t j, index_t value)
{
    assert((value & (~1UL)) == 0);
    map[j/64] = (map[j/64] & ~(1UL << (j%64))) | ((value&1) << (j%64));  
}

index_t bitget(index_t *map, index_t j)
{
    return (map[j/64]>>(j%64))&1UL;
}


/******************************************************************** Stack. */

typedef struct stack_node {
    index_t u;
    //index_t l;
    index_t t;
} stack_node_t;

typedef struct stack {
    index_t size; // size of stack
    index_t n; // number of elements
    stack_node_t *a;
}stk_t;

stk_t * stack_alloc(index_t size)
{
    stk_t *s = (stk_t *) malloc(sizeof(stk_t));
    s->size = size;
    s->n = 0;
    s->a = (stack_node_t *) malloc(s->size*sizeof(stack_node_t));
    return s;
}

void stack_free(stk_t *s)
{
    free(s->a);
    free(s);
}

void stack_push(stk_t *s, stack_node_t *e_in)
{
    assert(s->n < s->size);
    stack_node_t *e = s->a + s->n;
    e->u = e_in->u;
    //e->l = e_in->l;
    e->t = e_in->t;
    s->n++;
}

void stack_pop(stk_t *s, stack_node_t *e_out)
{
    assert(s->n > 0);
    s->n--;
    stack_node_t *e = s->a + s->n;
    e_out->u = e->u;
    //e_out->l = e->l;
    e_out->t = e->t;
}

void stack_top(stk_t *s, stack_node_t *e_out)
{
    assert(s->n >= 0);
    stack_node_t *e = s->a + s->n-1;
    e_out->u = e->u;
    //e_out->l = e->l;
    e_out->t = e->t;
}

void stack_empty(stk_t *s)
{
#ifdef DEBUG
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        e->u = UNDEFINED;
        //e.l = UNDEFINED;
        e->t = UNDEFINED;
    }
#endif
    s->n = 0;
}

void stack_get_vertices(stk_t *s, index_t *uu)
{
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        uu[i] = e->u;
    }
}

void stack_get_timestamps(stk_t *s, index_t *tt)
{
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        tt[i] = e->t;
    }
}

#ifdef DEBUG
void print_stack(stk_t *s)
{
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "print stack\n");
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "size: %ld\n", s->size);
    fprintf(stdout, "n: %ld\n", s->n);
    fprintf(stdout, "a: ");
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        fprintf(stdout, "[%ld, %ld, %ld]%s", 
                         e->u==UNDEFINED ? UNDEFINED : e->u+1, 
                         e->l, e->t, (i==s->n-1)?"\n":" ");
    }
    fprintf(stdout, "-----------------------------------------------\n");
}

void print_stacknode(stack_node_t *e)
{
    fprintf(stdout, "print stack-node: [%ld, %ld, %ld]\n", e->u, e->l, e->t);
}
#endif

/*************************************************** Random numbers and such. */

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand());
}

index_t randnum(index_t range)
{
    return (((index_t) rand())<<31)^((index_t) rand()) % range;
}

void randseq(index_t n, index_t range, index_t seed, index_t *a)
{
    ffprng_t base;
    FFPRNG_INIT(base, seed);
    index_t nt = num_threads();
    index_t block_size = n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t rs = (index_t) (rnd&0X7FFFFFFFFFFFFFFF);
            a[i] = rs%range;
        }
    }
}

// returns a number in range <min, max-1>
index_t randrange(index_t min, index_t max)
{
    return irand()%(max + 1 - min) + min;
}


void randshuffle_seq(index_t n, index_t *p, ffprng_t gen)
{
    for(index_t i = 0; i < n-1; i++) {
        ffprng_scalar_t rnd;
        FFPRNG_RAND(rnd, gen);
        index_t x = i+(rnd%(n-i));
        index_t t = p[x];
        p[x] = p[i];
        p[i] = t;
    }
}

void randperm(index_t n, index_t seed, index_t *p)
{
#ifdef BUILD_PARALLEL
    index_t nt = 64;
#else
    index_t nt = 1;
#endif
    index_t block_size = n/nt;
    index_t f[128][128];
    assert(nt < 128);

    ffprng_t base;
    FFPRNG_INIT(base, seed);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        for(index_t j = 0; j < nt; j++)
            f[t][j] = 0;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        ffprng_t gen;
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            f[t][bin]++;
        }
    }

    for(index_t bin = 0; bin < nt; bin++) {
        for(index_t t = 1; t < nt; t++) {
            f[0][bin] += f[t][bin];
        }
    }
    index_t run = 0;
    for(index_t j = 1; j <= nt; j++) {
        index_t fp = f[0][j-1];
        f[0][j-1] = run;
        run += fp;
    }
    f[0][nt] = run;

    FFPRNG_INIT(base, seed);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = 0;
        index_t stop = n-1;
        index_t pos = f[0][t];
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            if(bin == t)
                p[pos++] = i;
        }
        assert(pos == f[0][t+1]);
    }

    FFPRNG_INIT(base, (seed^0x9078563412EFDCABL));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t fwd, gen;
        index_t start = f[0][t];
        index_t stop = f[0][t+1]-1;
        index_t u;
        FFPRNG_FWD(fwd, (1234567890123456L*t), base);
        FFPRNG_RAND(u, fwd);
        FFPRNG_INIT(gen, u);
        randshuffle_seq(stop-start+1, p + start, gen);
    }
}

void rand_nums(index_t seed, index_t n, index_t *p)
{
#ifdef BUILD_PARALLEL
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        srand(seed+th);
        for(index_t i = start; i <= stop; i++)
            p[i] = rand();
    }
#else
    srand(seed);
    for(index_t i = 0; i < n; i++)
        p[i] = rand();
#endif
}

/***************************************************** (Parallel) prefix sum. */

index_t prefixsum(index_t n, index_t *a, index_t k)
{

#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = (stop-start+1)*k;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    index_t run = 0;
    for(index_t t = 1; t <= nt; t++) {
        index_t v = s[t-1];
        s[t-1] = run;
        run += v;
    }
    s[nt] = run;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t trun = s[t];
        for(index_t u = start; u <= stop; u++) {
            index_t tv = a[u];
            a[u] = trun;
            trun += tv + k;
        }
        assert(trun == s[t+1]);    
    }

#else

    index_t run = 0;
    for(index_t u = 0; u < n; u++) {
        index_t tv = a[u];
        a[u] = run;
        run += tv + k;
    }

#endif

    return run; 
}

/************************************************************* Parallel sum. */

index_t parallelsum(index_t n, index_t *a)
{
    index_t sum = 0;
#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = 0;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    for(index_t t = 0; t < nt; t++)
        sum += s[t];
#else
    for(index_t i = 0; i < n; i++) {
        sum += a[i];
    }
#endif
    return sum;
}

/********************************** Initialize an array with random scalars. */

void randinits_scalar(scalar_t *a, index_t s, ffprng_scalar_t seed) 
{
    ffprng_t base;
    FFPRNG_INIT(base, seed);
    index_t nt = num_threads();
    index_t block_size = s/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? s-1 : (start+block_size-1);
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            scalar_t rs = (scalar_t) rnd;           
            a[i] = rs;
        }
    }
}

/************************************************* Rudimentary graph builder. */

typedef struct 
{
    index_t is_directed;
    index_t num_vertices;
    index_t num_edges;
    index_t max_time;
    index_t edge_capacity;
    index_t *edges;
    index_t *colors;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC(sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE(was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    index_t i;
    graph_t *g = (graph_t *) MALLOC(sizeof(graph_t));
    g->is_directed   = 0; // default: undirected graph
    g->num_vertices  = n;
    g->num_edges     = 0;
    g->edge_capacity = 100;
    g->edges  = enlarge(3*g->edge_capacity, 0, (index_t *) 0);
    g->colors = (index_t *) MALLOC(sizeof(index_t)*n);
    for(i = 0; i < n; i++)
        g->colors[i] = UNDEFINED;
    return g;
}

void graph_free(graph_t *g)
{
    FREE(g->edges);
    FREE(g->colors);
    FREE(g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v, index_t t)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);
    assert(t>=0);
    //assert(t>=0 && t < g->max_time);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(6*g->edge_capacity, 3*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 3*g->num_edges;
    e[0] = u;
    e[1] = v;
    e[2] = t;
    g->num_edges++;
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    g->edges = enlarge(3*g->edge_capacity+3*cap, 3*g->edge_capacity, g->edges);
    index_t *e = g->edges + 3*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

void graph_set_color(graph_t *g, index_t u, index_t c)
{
    assert(u >= 0 && u < g->num_vertices && c >= 0);
    g->colors[u] = c;
}

void graph_set_is_directed(graph_t *g, index_t is_dir)
{
    assert(is_dir == 0 || is_dir == 1);
    g->is_directed = is_dir;
}

void graph_set_max_time(graph_t *g, index_t tmax)
{
    assert(tmax > 0);
    g->max_time = tmax;
}

#ifdef DEBUG
void print_graph(graph_t *g)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t tmax = g->max_time;
    fprintf(stdout, "p motif %ld %ld %ld\n", n, m, tmax);

    index_t *e = g->edges;
    for(index_t i = 0; i < 3*m; i+=3) {
        fprintf(stdout, "e %ld %ld %ld\n", 
                        e[i]+1, e[i+1]+1, e[i+2]+1);
    }

    index_t *c = g->colors;
    for(index_t i = 0; i < n; i++)
        fprintf(stdout, "n %ld %ld\n", i+1, c[i]+1);
}
#endif


/************************************* Basic motif query processing routines. */

struct temppathq_struct
{
    index_t     is_stub;
    index_t     n;
    index_t     k;
    index_t     tmax;
    index_t     *pos;
    index_t     *adj;
    index_t     nl;
    index_t     *l;  
    index_t     ns;
    shade_map_t *shade;
    index_t     *color;
};

typedef struct temppathq_struct temppathq_t;

void adjsort(index_t n, index_t *pos, index_t *adj)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        heapsort_indext(deg, adj + pu + 1);
    }
}

void temppathq_free(temppathq_t *q)
{
    if(!q->is_stub) {
        FREE(q->pos);
        FREE(q->adj);
        FREE(q->l);
        FREE(q->shade);
        FREE(q->color);
    }
    FREE(q);
}

#ifdef DEBUG
void print_temppathq(temppathq_t *root)
{
    index_t n       = root->n;
    index_t k       = root->k;
    index_t tmax    = root->tmax;
    index_t *pos    = root->pos;
    index_t *adj    = root->adj;
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "printing temppathq\n");
    fprintf(stdout, "is_stub = %ld\n", root->is_stub);
    fprintf(stdout, "n = %ld\n", n);
    fprintf(stdout, "k = %ld\n", k);
    fprintf(stdout, "tmax = %ld\n", tmax);
    fprintf(stdout, "pos\n");
    fprintf(stdout, "----\n ");
    for(index_t i = 0; i < n*tmax; i++) {
        fprintf(stdout, "%ld%s", pos[i], i%n==n-1 ? "\n ":" ");
    }

    fprintf(stdout, "adjacency list:\n");
    fprintf(stdout, "---------------\n");
    for(index_t t = 0; t < tmax; t++) {
        fprintf(stdout, "t: %ld\n", t+1);
        fprintf(stdout, "---------------\n");

        index_t *pos_t = pos + n*t;
        for(index_t u = 0; u < n; u++) {
            index_t pu = pos_t[u];
            index_t nu = adj[pu];
            index_t *adj_u = adj + pu + 1;
            fprintf(stdout, "%4ld:", u+1);
            for(index_t i = 0; i < nu; i++) {
                fprintf(stdout, " %4ld", adj_u[i]+1);
            }
            fprintf(stdout, "\n");
        }
    }

    index_t nl          = root->nl;
    index_t *l          = root->l;
    fprintf(stdout, "nl = %ld\n", nl);
    fprintf(stdout, "l:\n");
    for(index_t i = 0; i < nl; i++)
        fprintf(stdout, "%8ld : %8ld\n", nl, l[i]);

    index_t ns = root ->ns;
    shade_map_t *shade  = root->shade;
    fprintf(stdout, "ns : %ld\n", ns);
    fprintf(stdout, "shades:\n");
    for(index_t u = 0; u < n; u++)
        fprintf(stdout, "%10ld : 0x%08X\n", u+1, shade[u]);

    index_t *color = root->color;
    fprintf(stdout, "color:\n");
    for(index_t u = 0; u < n; u++)
        fprintf(stdout, "%10ld: %4ld\n", u+1, color[u]);
    fprintf(stdout, "-----------------------------------------------\n");
}

void print_array(const char *name, index_t n, index_t *a, index_t offset)
{
    fprintf(stdout, "%s (%ld):", name, n);
    for(index_t i = 0; i < n; i++) {
        fprintf(stdout, " %ld", a[i] == -1 ? -1 : a[i]+offset);
    }
    fprintf(stdout, "\n"); 
}
#endif

/******************************************************** Root query builder. */

// Query builder for directed graphs
//
temppathq_t *build_temppathq_dir(graph_t *g, index_t k, index_t *kk)
{
    push_memtrack();

    index_t n           = g->num_vertices;
    index_t m           = g->num_edges;
    index_t tmax        = g->max_time;
    index_t *pos        = alloc_idxtab(n*tmax);
    index_t *adj        = alloc_idxtab(n*tmax+2*m);
    index_t ns          = k;
    shade_map_t *shade  = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);
    index_t *color      = alloc_idxtab(n);

    temppathq_t *root = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    root->is_stub = 0;
    root->n       = g->num_vertices;
    root->k       = k;
    root->tmax    = tmax;
    root->pos     = pos;
    root->adj     = adj;
    root->nl      = 0;
    root->l       = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    root->ns      = ns;
    root->shade   = shade;
    root->color   = color;

    assert(tmax >= k-1);

    push_time();
    fprintf(stdout, "build query: ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            //index_t u = e[j];
            index_t v = e[j+1];
            index_t t = e[j+2];
            index_t *pos_t = (pos + (n*t));
            //if(start <= u && u <= stop) {
            //    // I am responsible for u, record adjacency to u
            //    pos_t[u]++;
            //}
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                pos_t[v]++;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        //index_t u = e[j];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t *pos_t = pos + n*t;
        //pos_t[u]++;
        pos_t[v]++;
    }
#endif

    index_t run = prefixsum(n*tmax, pos, 1);
    assert(run == (n*tmax+m));
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
            adj[pos[u]] = 0;

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t t = e[j+2];
            //if(start <= u && u <= stop) {
            //    // I am responsible for u, record adjacency to u
            //    index_t pu = pos[n*t+u];
            //    adj[pu + 1 + adj[pu]++] = v;
            //}
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                index_t pv = pos[n*t+v];
                adj[pv + 1 + adj[pv]++] = u;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t t = e[j+2];
        //index_t pu = pos[n*t+u];
        index_t pv = pos[n*t+v];       
        //adj[pu + 1 + adj[pu]++] = v;
        adj[pv + 1 + adj[pv]++] = u;
    }
#endif
    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    //print_temppathq(root);
    push_time();
    adjsort(n*tmax, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(g->colors[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
        //fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    }

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        color[u] = g->colors[u];

    time = pop_time();
    fprintf(stdout, "[shade: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

// Query builder for undirected graphs
//
temppathq_t *build_temppathq(graph_t *g, index_t k, index_t *kk)
{
    push_memtrack();

    index_t n           = g->num_vertices;
    index_t m           = g->num_edges;
    index_t tmax        = g->max_time;
    index_t *pos        = alloc_idxtab(n*tmax);
    index_t *adj        = alloc_idxtab(n*tmax+2*m);
    index_t ns          = k;
    shade_map_t *shade  = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);
    index_t *color      = alloc_idxtab(n);

    temppathq_t *root = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    root->is_stub = 0;
    root->n       = g->num_vertices;
    root->k       = k;
    root->tmax    = tmax;
    root->pos     = pos;
    root->adj     = adj;
    root->nl      = 0;
    root->l       = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    root->ns      = ns;
    root->shade   = shade;
    root->color   = color;

    assert(tmax >= k-1);

    push_time();
    fprintf(stdout, "build query: ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j];
            index_t v = e[j+1];
            index_t t = e[j+2];
            index_t *pos_t = (pos + (n*t));
            if(start <= u && u <= stop) {
                // I am responsible for u, record adjacency to u
                pos_t[u]++;
            }
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                pos_t[v]++;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t *pos_t = pos + n*t;
        pos_t[u]++;
        pos_t[v]++;
    }
#endif

    index_t run = prefixsum(n*tmax, pos, 1);
    assert(run == (n*tmax+2*m));
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++) {
            adj[pos[u]] = 0;
    }

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t t = e[j+2];
            if(start <= u && u <= stop) {
                // I am responsible for u, record adjacency to u
                index_t pu = pos[n*t+u];
                adj[pu + 1 + adj[pu]++] = v;
            }
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                index_t pv = pos[n*t+v];
                adj[pv + 1 + adj[pv]++] = u;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t pu = pos[n*t+u];
        index_t pv = pos[n*t+v];       
        adj[pu + 1 + adj[pu]++] = v;
        adj[pv + 1 + adj[pv]++] = u;
    }
#endif
    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    //print_temppathq(root);
    push_time();
    adjsort(n*tmax, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(g->colors[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
//        fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    }

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        color[u] = g->colors[u];

    time = pop_time();
    fprintf(stdout, "[shade: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

void query_pre_mk1(temppathq_t *in, temppathq_t **out_q, index_t **out_map)
{
    push_memtrack();

    index_t nt = num_threads();
    index_t i_n          = in->n;
    index_t k            = in->k;
    index_t tmax         = in->tmax;
    index_t *i_pos       = in->pos;
    index_t *i_adj       = in->adj;
    index_t ns           = in->ns;
    shade_map_t *i_shade = in->shade;
    index_t *i_color     = in->color;

    push_time();
    fprintf(stdout, "query pre [1]: ");
    fflush(stdout);

    push_time();
    // input-to-output vertex map
    index_t *v_map_i2o   = (index_t *) MALLOC(sizeof(index_t)*i_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++)
        v_map_i2o[u] = UNDEFINED;

    index_t v_cnt = 0;
#ifdef BUILD_PARALLEL
    // parallely construct input-to-output vertex map
    index_t block_size = i_n/nt;
    index_t t_vcnt[nt];

#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
        t_vcnt[th] = 0;
        for(index_t u = start; u <= stop; u++) {
            if(i_shade[u])
                v_map_i2o[u] = t_vcnt[th]++;
        }
    }
  
    // prefix sum
    for(index_t th = 1; th < nt; th++)
        t_vcnt[th] += t_vcnt[th-1];

#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
        index_t tsum = (th==0 ? 0 : t_vcnt[th-1]);
        for(index_t u = start; u <= stop; u++) {
            if(i_shade[u])
                v_map_i2o[u] += tsum;
        }
    }
    v_cnt = t_vcnt[nt-1];

#else
    // serially construct input-to-output vertex map
    for(index_t u = 0; u < i_n; u++) {
        if(i_shade[u])
            v_map_i2o[u] = v_cnt++;
    }
#endif

    // output-to-input vertex map 
    // required to reconstruct solution in original graph
    index_t o_n = v_cnt;
    index_t *v_map_o2i = (index_t *) MALLOC(sizeof(index_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            v_map_o2i[o_u] = u;
    }

    fprintf(stdout, "[map: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output position list
    index_t *o_pos = alloc_idxtab(o_n*tmax);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u =  v_map_i2o[u];
                if(o_u == UNDEFINED) continue;
                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    o_pos_t[o_u]++;
                }
            }
        }
    }

    index_t o_m   = parallelsum(o_n*tmax, o_pos);
    index_t run   = prefixsum(o_n*tmax, o_pos, 1);
    assert(run == (o_n*tmax+o_m));

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output adjacency list
    index_t *o_adj = alloc_idxtab(o_n*tmax + o_m);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_adj[o_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u = v_map_i2o[u];
                if(o_u == UNDEFINED) continue;

                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                index_t o_pu = o_pos_t[o_u];
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    
                    o_adj[o_pu + 1 + o_adj[o_pu]++] = o_v;
                }
            }
        }
    }

    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output shade map
    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_shade[o_u] = i_shade[u];
    }

    // output color
    index_t *o_color = alloc_idxtab(o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_color[o_u] = i_color[u];
    }

    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fflush(stdout);

    temppathq_t *out = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    out->is_stub     = 0;
    out->n           = o_n;
    out->k           = k;
    out->tmax        = tmax;
    out->pos         = o_pos;
    out->adj         = o_adj;
    out->nl          = 0;
    out->l           = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns          = ns;
    out->shade       = o_shade;
    out->color       = o_color;

    *out_q           = out;
    *out_map         = v_map_o2i;

    FREE(v_map_i2o);

    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);
}

/****************************************************** Input reader (ASCII). */

void skipws(FILE *in)
{
    int c;
    do {
        c = fgetc(in);
        if(c == '#') {
            do {
                c = fgetc(in);
            } while(c != EOF && c != '\n');
        }
    } while(c != EOF && isspace(c));
    if(c != EOF)
        ungetc(c, in);
}

#define CMD_NOP             0
#define CMD_TEST_UNIQUE     1
#define CMD_TEST_COUNT      2
#define CMD_RUN_ORACLE      3
#define CMD_LIST_FIRST      4
#define CMD_LIST_ALL        5
#define CMD_BASE_TEMPPATH   6
#define CMD_BASE_PATHMOTIF  7
#define CMD_BASE_DFS        8

const char *cmd_legend[] = { "no operation", 
                       "test unique",
                       "test count",
                       "run oracle",
                       "list first",
                       "list all",
                       "baseline temppath",
                       "baseline pathmotif",
                       "baseline dfs"};

void reader_ascii(FILE *in, 
                  graph_t **g_out, index_t *k_out, index_t **kk_out, 
                  index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t n         = 0;
    index_t m         = 0;
    index_t tmax      = 0;
    index_t is_dir    = 0;
    graph_t *g        = (graph_t *) 0;
    index_t *kk       = (index_t *) 0;
    index_t cmd       = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    index_t i, j, d, k, t;
    skipws(in);
    while(!feof(in)) {
        skipws(in);
        int c = fgetc(in);
        switch(c) {
        case 'p':
            if(g != (graph_t *) 0)
                ERROR("duplicate parameter line");
            skipws(in);
            if(fscanf(in, "motif %ld %ld %ld %ld", &n, &m, &tmax, &is_dir) != 4)
                ERROR("invalid parameter line");
            if(n <= 0 || m < 0 ) {
                ERROR("invalid input parameters (n = %ld, m = %ld, tmax = %ld)",
                       n, m, tmax);
            }
            g = graph_alloc(n);
            graph_set_is_directed(g, is_dir);
            graph_set_max_time(g, tmax);
            break;
        case 'e':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before edges");
            skipws(in);
            if(fscanf(in, "%ld %ld %ld", &i, &j, &t) != 3)
                ERROR("invalid edge line");
            //if(i < 1 || i > n || j < 1 || j > n || t < 1 || t > tmax) {
            //    ERROR("invalid edge (i = %ld, j = %ld t = %ld with n = %ld, tmax = %ld)", 
            //          i, j, t, n, tmax);
            //}
            graph_add_edge(g, i-1, j-1, t-1);
            break;
        case 'n':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before vertex colors");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &d) != 2)
                ERROR("invalid color line");
            if(i < 1 || i > n || d < 1)
                ERROR("invalid color line (i = %ld, d = %ld with n = %ld)", 
                      i, d, n);
            graph_set_color(g, i-1, d-1);
            break;
        case 'k':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before motif");
            skipws(in);
            if(fscanf(in, "%ld", &k) != 1)
                ERROR("invalid motif line");
            if(k < 1 || k > n)
                ERROR("invalid motif line (k = %ld with n = %d)", k, n);
            kk = alloc_idxtab(k);
            for(index_t u = 0; u < k; u++) {
                skipws(in);
                if(fscanf(in, "%ld", &i) != 1)
                    ERROR("error parsing motif line");
                if(i < 1)
                    ERROR("invalid color on motif line (i = %ld)", i);
                kk[u] = i-1;
            }
            break;
        case 't':
            if(g == (graph_t *) 0 || kk == (index_t *) 0)
                ERROR("parameter and motif lines must be given before test");
            skipws(in);
            {
                char cmdstr[128];
                if(fscanf(in, "%100s", cmdstr) != 1)
                    ERROR("invalid test command");
                if(!strcmp(cmdstr, "unique")) {
                    cmd_args = alloc_idxtab(k);
                    for(index_t u = 0; u < k; u++) {
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 1 || i > n)
                            ERROR("invalid test line entry (i = %ld)", i);
                        cmd_args[u] = i-1;
                    }
                    heapsort_indext(k, cmd_args);
                    for(index_t u = 1; u < k; u++)
                        if(cmd_args[u-1] >= cmd_args[u])
                            ERROR("test line contains duplicate entries");
                    cmd = CMD_TEST_UNIQUE;
                } else {
                    if(!strcmp(cmdstr, "count")) {
                        cmd_args = alloc_idxtab(1);
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 0)
                            ERROR("count on test line cannot be negative");
                        cmd = CMD_TEST_COUNT;
                        cmd_args[0] = i;
                    } else {
                        ERROR("unrecognized test command \"%s\"", cmdstr);
                    }
                }
            }
            break;
        case EOF:
            break;
        default:
            ERROR("parse error");
        }
    }

    if(g == (graph_t *) 0)
        ERROR("no graph given in input");
    if(kk == (index_t *) 0)
        ERROR("no motif given in input");

    for(index_t i = 0; i < n; i++) {
        if(g->colors[i] == -1)
            ERROR("no color assigned to vertex i = %ld", i);
    }
    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld, t = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            g->max_time,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");    
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}

/******************************************************************************/

void get_subgraph(index_t *kk, temppathq_t *in, temppathq_t **out_q, 
                  index_t **out_map)
{
    push_memtrack();

    index_t nt           = num_threads();
    index_t i_n          = in->n;
    index_t k            = in->k;
    index_t tmax         = in->tmax;
    index_t *i_pos       = in->pos;
    index_t *i_adj       = in->adj;
    index_t ns           = in->ns;
    shade_map_t *i_shade = in->shade;
    index_t *i_color     = in->color;

    // output graph
    index_t o_n = k;

    push_time();
    fprintf(stdout, "get_subgraph: ");
    fflush(stdout);

    //shellsort(n, kk);

    push_time();
    // input-to-output vertex map
    index_t *v_map_i2o   = (index_t *) MALLOC(sizeof(index_t)*i_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++)
        v_map_i2o[u] = UNDEFINED;

    // serially construct input-to-output vertex map
    for(index_t i = 0; i < k; i++)
        v_map_i2o[kk[i]] = i;

    // output-to-input vertex map 
    // required to reconstruct solution in original graph
    index_t *v_map_o2i = (index_t *) MALLOC(sizeof(index_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < o_n; i++) {
        v_map_o2i[i] = kk[i];
    }

    fprintf(stdout, "[map: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output position list
    index_t *o_pos = alloc_idxtab(o_n*tmax);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_pos[u] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u =  v_map_i2o[u];
                if(o_u == UNDEFINED) continue;
                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    o_pos_t[o_u]++;
                }
            }
        }
    }

    index_t o_m   = parallelsum(o_n*tmax, o_pos);
    index_t run   = prefixsum(o_n*tmax, o_pos, 1);
    assert(run == (o_n*tmax+o_m));

    fprintf(stdout, "[pos: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output adjacency list
    index_t *o_adj = alloc_idxtab(o_n*tmax + o_m);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < o_n*tmax; u++)
        o_adj[o_pos[u]] = 0;

    for(index_t t = 0; t < tmax; t++) {
        index_t *o_pos_t = o_pos + o_n*t;
        index_t *i_pos_t = i_pos + i_n*t;
        index_t block_size = i_n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t th = 0; th < nt; th++) {
            index_t start = th*block_size;
            index_t stop = (th == nt-1) ? i_n-1 : (start+block_size-1);
            for(index_t u = start; u <= stop; u++) {
                index_t o_u = v_map_i2o[u];
                if(o_u == UNDEFINED) continue;

                index_t i_pu = i_pos_t[u];
                index_t i_nu = i_adj[i_pu];
                index_t *i_adj_u = i_adj + i_pu;
                index_t o_pu = o_pos_t[o_u];
                for(index_t j = 1; j <= i_nu; j++) {
                    index_t v = i_adj_u[j];
                    index_t o_v = v_map_i2o[v];
                    if(o_v == UNDEFINED) continue;
                    
                    o_adj[o_pu + 1 + o_adj[o_pu]++] = o_v;
                }
            }
        }
    }
    fprintf(stdout, "[adj: %.2lf ms] ", pop_time());
    fflush(stdout);
    push_time();

    // output shade map
    shade_map_t *o_shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_shade[o_u] = i_shade[u];
    }

    index_t *o_color = alloc_idxtab(o_n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < i_n; u++) {
        index_t o_u = v_map_i2o[u];
        if(o_u != UNDEFINED)
            o_color[o_u] = i_color[u];
    }

    fprintf(stdout, "[shade: %.2lf ms] ", pop_time());
    fflush(stdout);

    temppathq_t *out = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    out->is_stub     = 0;
    out->n           = o_n;
    out->k           = k;
    out->tmax        = tmax;
    out->pos         = o_pos;
    out->adj         = o_adj;
    out->nl          = 0;
    out->l           = (index_t *) MALLOC(sizeof(index_t)*out->nl);
    out->ns          = ns;
    out->shade       = o_shade;
    out->color       = o_color;

    *out_q           = out;
    *out_map         = v_map_o2i;

    FREE(v_map_i2o);

    fprintf(stdout, "done. [%.2lf ms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);
}

/*****************************************************************************/

index_t temp_dfs(index_t n, index_t k, index_t t_max, index_t *pos, 
                 index_t *adj, index_t *color, index_t *kk_in, 
                 index_t *in_stack, stk_t *s, index_t *uu_out, 
                 index_t *tt_out, index_t *t_opt)
{
    if(s->n >= k) {
        // reached depth k
        assert(s->n <= k);

        // allocate memory
        index_t *uu_sol = (index_t *) malloc(k*sizeof(index_t));
        index_t *kk_sol = (index_t *) malloc(k*sizeof(index_t));
        index_t *tt_sol = (index_t *) malloc(k*sizeof(index_t));

        // get vertices in stack
        stack_get_vertices(s, uu_sol);
        stack_get_timestamps(s, tt_sol);
        // get vertex colors
        for(index_t i = 0; i < k; i++)
            kk_sol[i] = color[uu_sol[i]];
		shellsort(k, kk_sol);

        // check if colors match
        index_t is_motif = 1;
        for(index_t i = 0; i < k; i++) {
            if(kk_sol[i] != kk_in[i]) {
				is_motif = 0;
				break;
			}
        }

        // match found
        if(is_motif) {
            stack_node_t e;
            stack_top(s, &e);
            if(*t_opt > e.t) {
                // copy solution vertices
                for(index_t i = 0; i < k; i++)
                    uu_out[i] = uu_sol[i];
                // copy solution timestamps
                for(index_t i = 0; i < k; i++)
                    tt_out[i] = tt_sol[i];
                *t_opt = e.t;
            }
        }

        // free memory
        free(uu_sol);
        free(kk_sol);
        free(tt_sol);
        return 1;
    } else {
        // get stack-top
        stack_node_t e;
        stack_top(s, &e);
        index_t u    = e.u;
        //index_t l    = e.l;
        index_t t_min = e.t;

        // proceed with temporal DFS
        for(index_t t = t_min; t < t_max; t++) {
            index_t *pos_t = pos + t*n;
            index_t pu = pos_t[u];
            index_t nu = adj[pu];
            if(nu == 0) continue;

            index_t *adj_u = adj + pu;
            for(index_t i = 1; i <= nu; i++) {
                index_t v = adj_u[i];
                if(in_stack[v]) continue;

                stack_node_t e;
                e.u = v;
                //e.l = l+1;
                e.t = t+1;
                stack_push(s, &e);
                in_stack[v] = 1;

                // recursive call to depth k
                temp_dfs(n, k, t_max, pos, adj, color, kk_in, in_stack, s,
                         uu_out, tt_out, t_opt);

                stack_pop(s, &e);
                in_stack[v] = 0;
            }
        }
    }
    return 1; // not found
}

index_t baseline_dfs(index_t seed, temppathq_t *root, index_t *kk)
{
    push_time();
    // thread count
    index_t nt      = num_threads();
    // init
    index_t n       = root->n;
    index_t k       = root->k;
    index_t t_max   = root->tmax;
    index_t *pos    = root->pos;
    index_t *adj    = root->adj;
    index_t *color  = root->color;

    // allocate memory
    index_t *uu_sol_nt      = alloc_idxtab(nt*k);
    index_t *tt_sol_nt      = alloc_idxtab(nt*k);
    index_t *in_stack_nt    = alloc_idxtab(nt*n);
    index_t *t_opt_nt       = alloc_idxtab(nt);

    // initialise and time-it
    push_time();
    shellsort(k, kk);
    index_t *v_seq = alloc_idxtab(n);
    randperm(n, seed, v_seq);
    double init_time = pop_time();
    push_time();

    index_t block_size = n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop  = (th == nt-1) ? n-1 : (start+block_size-1);
        stk_t *s      = stack_alloc(k); // allocate stack
        // memory for each thread
        index_t *in_stack   = in_stack_nt + th*n;
        index_t *uu_sol     = uu_sol_nt + th*k;
        index_t *tt_sol     = tt_sol_nt + th*k;
        index_t *t_opt      = t_opt_nt + th;
        // initialise optimal-time to infinity
        *t_opt              = MATH_INF;

        // each thread handle a set of vertices
        for(index_t j = start; j <= stop; j++) {
            // initialise stack to empty
            for(index_t i = 0; i < n; i++)
                in_stack[i] = 0;
            
            index_t u = v_seq[j];
            stack_node_t e;
            e.u = u;
            //e.l = 1;
            e.t = 0;
            stack_push(s, &e);
            in_stack[u] = 1;

            // perform temporal-DFS
            temp_dfs(n, k, t_max, pos, adj, color, kk, in_stack, s, uu_sol,
                     tt_sol, t_opt);

            // empty stack
            stack_empty(s);
        }
        stack_free(s);
    }

    // obtain global optimum using optimal values from each thread
    index_t *uu_opt = (index_t *) MALLOC(k*sizeof(index_t));
    index_t *tt_opt = (index_t *) MALLOC(k*sizeof(index_t));
    index_t t_opt = MATH_INF;
    for(index_t th = 0; th < nt; th++) {
        index_t *uu_sol = uu_sol_nt + th*k;
        index_t *tt_sol = tt_sol_nt + th*k;
        if(t_opt > t_opt_nt[th]) {
            t_opt = t_opt_nt[th];
            for(index_t i = 0; i < k; i++)
                uu_opt[i] = uu_sol[i];
            for(index_t i = 0; i < k; i++)
                tt_opt[i] = tt_sol[i];
        }
    }
    double dfs_time = pop_time();

    // output solution
    index_t found = 0;
    if(t_opt != MATH_INF) {
        found = 1;
        fprintf(stdout, "solution [%ld, %.2lfms]: ", t_opt, dfs_time);
        for(index_t i = 0; i < k-1; i++) {
            index_t u = uu_opt[i];
            index_t v = uu_opt[i+1];
            index_t t = tt_opt[i+1];
            fprintf(stdout, "[%ld, %ld, %ld]%s", u+1, v+1, t, i==k-2?"\n":" ");
        }
    }

    FREE(in_stack_nt);
    FREE(uu_sol_nt);
    FREE(tt_sol_nt);
    FREE(t_opt_nt);
    FREE(tt_opt);
    FREE(uu_opt);
    FREE(v_seq);

    fprintf(stdout, "baseline [dfs]: [init: %.2lf ms] [dfs: %.2lf ms] done."
                    " [%.2lf ms] -- %s\n"
                    ,init_time, dfs_time, pop_time(), found?"true":"false");
    return found;
}

/*****************************************************************************/

static void random_tempwalk(index_t seed, index_t n, index_t k, index_t tmax,
                            index_t *pos, index_t *adj, index_t *uu_sol, 
                            index_t *tt_sol)
{
    for(index_t i = 0; i < k; i++)
        uu_sol[i] = UNDEFINED;

    for(index_t i = 0; i < k; i++)
        tt_sol[i] = UNDEFINED;

    srand(seed);
    index_t s = irand()%n; // pick a random start vertex
    uu_sol[0] = s; // initialise walk with start vertex
    tt_sol[0] = 0; // always start with 0 timestamp

    for(index_t l = 0; l < k-1; l++) {
        index_t u  = uu_sol[l]; 
        index_t tu = tt_sol[l];
        if(tu >= tmax-1) break;

        index_t t  = randrange(tu+1, tmax-1);
        //assert(t>tu && t<tmax);

        index_t *pos_t  = pos + t*n;
        index_t pu      = pos_t[u];
        index_t nu      = adj[pu];
        if(nu == 0) break;

        index_t *adj_u  = adj + pu + 1;
        index_t i       = randrange(0, nu-1);
        index_t v       = adj_u[i];

        uu_sol[l+1] = v;
        tt_sol[l+1] = t;
    }
}


index_t baseline_randwalk_path(index_t max_itr, temppathq_t *root, index_t *kk, 
                           index_t **uu_out, index_t **tt_out)
{
    push_time();
    push_memtrack();

    index_t n           = root->n;
    index_t k           = root->k;
    index_t tmax        = root->tmax;
    index_t *pos        = root->pos;
    index_t *adj        = root->adj;
    index_t *uu_sol     = alloc_idxtab(k);
    index_t *tt_sol     = alloc_idxtab(k);
    index_t path_found  = 0;

    for(index_t itr = 0; itr < max_itr; itr++) {
        push_time();
        index_t seed = irand();
        random_tempwalk(seed, n, k, tmax, pos, adj, uu_sol, tt_sol);

        // check if the walk is temporal
        index_t is_temp = 1;
        for(index_t i = 0; i < k-1; i++) {
            if(tt_sol[i] >= tt_sol[i+1]) {
                is_temp= 0;
                break;
            }
        }

        if(!is_temp) {
            fprintf(stdout, "%10ld : [%7ld %.4lfms] [not-path]\n", 
                            itr+1, uu_sol[0]+1, pop_time());
            fflush(stdout);
            continue;
        }

        // check if the walk is a path
        // sort and check if any two consecutive vertices are same
        index_t sol_temp[k];
        for(index_t i = 0; i < k; i++)
            sol_temp[i] = uu_sol[i];

        shellsort(k, sol_temp);
        index_t is_path = 1;
        for(index_t i = 0; i < k-1; i++) {
            if(sol_temp[i] == UNDEFINED || (sol_temp[i] == sol_temp[i+1])) {
                is_path = 0;
                break;
            }
        }

        if(!is_path) {
            fprintf(stdout, "%10ld : [%7ld %.4lfms] [not-path]\n", 
                            itr+1, uu_sol[0]+1, pop_time());
            fflush(stdout);
            continue;
        } else {
            path_found = 1;
            fprintf(stdout, "%10ld : [%7ld %.4lfms] [path-found]\n", 
                            itr+1, uu_sol[0]+1, pop_time());
            break;
        }
    }

    *uu_out = uu_sol;
    *tt_out = tt_sol;

    fprintf(stdout, "baseline-temppath: [%.4lfms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return path_found;
}

index_t baseline_randwalk_motif(index_t seed, index_t max_itr, 
                                temppathq_t *root, index_t *kk, 
                                index_t *uu_out, index_t *tt_out)
{
    push_time();
    push_memtrack();

    index_t nt          = num_threads();
    index_t n           = root->n;
    index_t k           = root->k;
    index_t tmax        = root->tmax;
    index_t *pos        = root->pos;
    index_t *adj        = root->adj;
    index_t *color      = root->color;
    index_t *uu_tmp_nt  = alloc_idxtab(k*nt);
    index_t *tt_tmp_nt  = alloc_idxtab(k*nt);
    index_t *kk_tmp_nt  = alloc_idxtab(k*nt);
    index_t *rand_seeds = alloc_idxtab(max_itr);

    // memory to store one solution per thread
    index_t *uu_sol_nt      = alloc_idxtab(k*nt);
    index_t *tt_sol_nt      = alloc_idxtab(k*nt);
    index_t *kk_sol_nt      = alloc_idxtab(k*nt);
    index_t *t_opt_nt       = alloc_idxtab(nt);

    shellsort(k, kk);
    // random seed for each iteration
    rand_nums(seed, max_itr, rand_seeds);

    index_t block_size      = max_itr/nt;
    index_t th_id           = UNDEFINED;
    volatile index_t found  = 0;
#ifdef BUILD_PARALLEL
#pragma omp parallel for shared(found)
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = 0;
        index_t stop  = (th == nt-1) ? max_itr-1 : (start+block_size-1);

        index_t *uu_tmp = uu_tmp_nt + th*k;
        index_t *tt_tmp = tt_tmp_nt + th*k;
        index_t *kk_tmp = kk_tmp_nt + th*k;

        index_t *uu_sol = uu_sol_nt + th*k;
        index_t *tt_sol = tt_sol_nt + th*k;
        index_t *kk_sol = kk_sol_nt + th*k;
        index_t *t_opt  = t_opt_nt + th;

        t_opt = MATH_INF; // initialise to MAX TIME

        for(index_t itr = start; itr <= stop; itr++) {
            //push_time();
            if(found) break;

            index_t seed_itr = rand_seeds[itr];
            random_tempwalk(seed_itr, n, k, tmax, pos, adj, uu_tmp, tt_tmp);

            // check if the walk is temporal
            index_t is_temp = 1;
            for(index_t i = 0; i < k-1; i++) {
                if(tt_tmp[i] >= tt_tmp[i+1]) {
                    is_temp= 0;
                    break;
                }
            }

            // continue, if walk is not temporal
            if(!is_temp) continue;

            // check if the walk is a path
            // sort and check if any two consecutive vertices are same
            index_t sol_temp[k];
            for(index_t i = 0; i < k; i++)
                sol_temp[i] = uu_tmp[i];

            shellsort(k, sol_temp);
            index_t is_path = 1;
            for(index_t i = 0; i < k-1; i++) {
                if(sol_temp[i] == UNDEFINED || (sol_temp[i] == sol_temp[i+1])) {
                    is_path = 0;
                    break;
                }
            }

            // continue, if walk is not a path
            if(!is_path) continue;

            // check if the colors in multiset match with path
            for(index_t i = 0; i < k; i++) {
                kk_tmp[i] = color[uu_tmp[i]];
                //kk_tmp[i] = __builtin_ffs(shade[uu_tmp[i]]);
            }

            // sort colors of math
            shellsort(k, kk_tmp);

            // check if the colors match
            index_t kk_match = 1;
            for(index_t i = 0; i < k; i++) {
                if(kk[i] != kk_tmp[i]) {
                    kk_match = 0;
                    break;
                }
            }

            // continue, if colors do not match
            if(!kk_match) continue;

            // found a motif match
            if(is_path && kk_match) {
                found = 1;
                th_id = th;
                continue;

                // TODO: update solution if the max-time is less than the
                // current solution
            }
        }
    }

    if(found) {
        assert(th_id != UNDEFINED);
        index_t *uu_tmp = uu_tmp_nt + k*th_id;
        index_t *tt_tmp = kk_tmp_nt + k*th_id;
        for(index_t i = 0; i < k; i++) 
            uu_out[i] = uu_tmp[i];

        for(index_t i = 0; i < k; i++)
            tt_out[i] = tt_tmp[i];
    }

    FREE(uu_tmp_nt);
    FREE(tt_tmp_nt);
    FREE(kk_tmp_nt);

    FREE(uu_sol_nt);
    FREE(tt_sol_nt);
    FREE(kk_sol_nt);
    FREE(t_opt_nt);

    FREE(color);
    FREE(rand_seeds);
    fprintf(stdout, "baseline-temppath: [%.4lfms] ", pop_time());
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return found;
}

/******************************************************* Program entry point. */

#define PRE_NOP 0
#define PRE_MK1 1

int main(int argc, char **argv)
{
    push_time();
    push_memtrack();
    
    index_t precomp    = PRE_NOP;
    index_t arg_cmd    = CMD_NOP;
    index_t max_itr    = UNDEFINED;
    index_t have_seed  = 0;
    index_t have_input = 0;
    index_t seed       = 123456789;
    char *filename     = (char *) 0;
    for(index_t f = 1; f < argc; f++) {
        if(argv[f][0] == '-') {
            if(!strcmp(argv[f], "-bin")) {
                flag_bin_input = 1;
            }
            if(!strcmp(argv[f], "-ascii")) {
                flag_bin_input = 0;
            }
            if(!strcmp(argv[f], "-temppath")) {
                arg_cmd = CMD_BASE_TEMPPATH;
            }
            if(!strcmp(argv[f], "-pathmotif")) {
                arg_cmd = CMD_BASE_PATHMOTIF;
            }
            if(!strcmp(argv[f], "-dfs")) {
                arg_cmd = CMD_BASE_DFS;
            }
            if(!strcmp(argv[f], "-all")) {
                arg_cmd = CMD_LIST_ALL;
            }
            if(!strcmp(argv[f], "-max-itr")) {
                if(f == argc-1)
                    ERROR("maximum iterations argument missing from command line");
                max_itr = atol(argv[++f]);
            }
            if(!strcmp(argv[f], "-pre")) {
                if(f == argc -1)
                    ERROR("preprocessing argument missing from command line");
                precomp = atol(argv[++f]);
            }
            if(!strcmp(argv[f], "-seed")) {
                if(f == argc - 1)
                    ERROR("random seed missing from command line");
                seed = atol(argv[++f]);
                have_seed = 1;
            }
            if(!strcmp(argv[f], "-in")) {
                if(f == argc - 1)
                    ERROR("input file missing from command line");
                filename = argv[++f];
                have_input = 1;
            }
        }
    }

    fprintf(stdout, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stdout, " %s", argv[f]);
    fprintf(stdout, "\n");

    if(have_seed == 0) {
        fprintf(stdout, 
                "no random seed given, defaulting to %ld\n", seed);
    }
    fprintf(stdout, "random seed = %ld\n", seed);
   
    if(max_itr == UNDEFINED) {
        max_itr = 10000;
        fprintf(stdout, 
                "no max iterations give, defaulting to %ld\n", max_itr);
    }
    fprintf(stdout, "max iterations = %ld\n", max_itr);

    FILE *in = stdin;
    if(have_input) {
        in = fopen(filename, "r"); 
        if(in == NULL)
            ERROR("unable to open file '%s'", filename);
    } else {
        fprintf(stdout, "no input file specified, defaulting to stdin\n");
    }

    fflush(stdout);
        
    srand(seed); 
    graph_t *g;
    index_t k;
    index_t *kk;
    index_t input_cmd;
    index_t *cmd_args;

    // read graph
    reader_ascii(in, &g, &k, &kk, &input_cmd, &cmd_args);

    index_t cmd = input_cmd;  // by default execute command in input stream
    if(arg_cmd != CMD_NOP)
        cmd = arg_cmd;        // override command in input stream

    // build root query
    //index_t is_dir = 0;
    temppathq_t *root = (temppathq_t *) 0;
    if(g->is_directed) {
        //is_dir = 1;
        root = build_temppathq_dir(g, k, kk); 
    } else {
        root = build_temppathq(g, k, kk);
    }
    graph_free(g); // free graph

    push_time();
    // preprocess query and time it
    push_time();
    index_t *v_map1;
    switch(precomp) {
    case PRE_NOP:
    {
        // no precomputation
        fprintf(stdout, "no preprocessing, default execution\n");
        break;
    }
    case PRE_MK1:
    {
        // preprocess: remove vertices with no matching colors
        temppathq_t *root_pre;
        query_pre_mk1(root, &root_pre, &v_map1);
        temppathq_free(root);
        root = root_pre;
        //FREE(v_map1);

        // preprocessed graph statistics
        index_t o_n    = root->n;
        index_t tmax   = root->tmax;
        index_t *o_pos = root->pos;
        index_t *o_adj = root->adj;
        index_t o_m = (o_pos[o_n*(tmax-1) + o_n-1] + 
                      o_adj[o_pos[o_n*(tmax-1) + o_n-1]] - (o_n*tmax) + 1)/2;
        fprintf(stdout, "output pre [1]: n = %ld, m = %ld, k = %ld \n", 
                         o_n, o_m, k);
        fflush(stdout);
        break;
    }
    default:
        break;
    }

    double precomp_time = pop_time();
    push_time();

    fprintf(stdout, "command: %s\n", cmd_legend[cmd]);
    fflush(stdout);

    // execute command
    switch(cmd) {
    case CMD_NOP:
    {
        temppathq_free(root);
        break;
    }
    case CMD_BASE_TEMPPATH:
    {
        index_t *uu;
        index_t *tt;
        baseline_randwalk_path(max_itr, root, kk, &uu, &tt);

        fprintf(stdout, "found [%ld]: ", k);
        for(index_t i = 0; i < k-1; i++) {
            fprintf(stdout, "[%ld %ld %ld]%s", 
                            uu[i]+1, uu[i+1]+1, tt[i+1]+1, i == k-2 ? "\n" : " ");
        }
        fprintf(stdout, "max-time: %ld\n", tt[k-1]+1);

        FREE(uu);
        FREE(tt);
        temppathq_free(root);
        break;
    }
    case CMD_BASE_PATHMOTIF:
    {
        index_t *uu = alloc_idxtab(k);
        index_t *tt = alloc_idxtab(k);
        
        if(baseline_randwalk_motif(seed, max_itr, root, kk, uu, tt)) {
            fprintf(stdout, "solution [found, %ld]: ", k);
            for(index_t i = 0; i < k-1; i++) {
                fprintf(stdout, "[%ld %ld %ld]%s", 
                                uu[i]+1, uu[i+1]+1, tt[i+1]+1, i == k-2 ? "\n" : " ");
            }
            fprintf(stdout, "max-time: %ld\n", tt[k-1]+1);
        } else {
            fprintf(stdout, "solution [not-found, %ld]: \n", k);
        }

        FREE(uu);
        FREE(tt);
        temppathq_free(root);
        break;
    }
    case CMD_BASE_DFS:
    {
        baseline_dfs(seed, root, kk);
        temppathq_free(root);
        break;
    }
    default:
        assert(0);
        break;
    }

    // free vertex map
    if(precomp == PRE_MK1)
        FREE(v_map1);

    FREE(kk);
    
    double cmd_time = pop_time();
    double time = pop_time();
    fprintf(stdout, "command done [%.2lf ms %.2lf ms %.2lf ms %.2lf ms]\n", 
                    precomp_time, cmd_time, time, time);
    if(input_cmd != CMD_NOP)
        FREE(cmd_args);

    time = pop_time();
    fprintf(stdout, "grand total [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, "\n");
    fprintf(stdout, "host: %s\n", sysdep_hostname());
    fprintf(stdout, 
            "build: %s\n",
#ifdef BUILD_PARALLEL
            "multithreaded"
#else
            "single thread"
#endif
    );
    fprintf(stdout, 
            "compiler: gcc %d.%d.%d\n",
            __GNUC__,
            __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
    fflush(stdout);
    assert(malloc_balance == 0);
    assert(memtrack_stack_top < 0);
    return 0;
}
