#include "exprParser.H"

#include <array>
#include <cctype>
#include <cstdlib>

namespace tedp
{
namespace expr
{

namespace
{

const std::array<std::string, N_STATE_VARS> stateVars
{
    "I1", "I2", "I3", "I4", "I5", "Ret", "F1", "PoE",
    "sigmaSL", "phiDkPk", "phiDkCk", "phik", "ReOmega",
    "Gp", "Gk", "Apk", "Psn", "Ksn", "Rf", "Rw"
};

struct Token
{
    enum class Kind { Num, Name, Op, End };
    Kind kind = Kind::End;
    std::string text;
    double value = 0.0;
};

class Lexer
{
    const std::string& s_;
    std::size_t pos_ = 0;

public:
    explicit Lexer(const std::string& s) : s_(s) {}

    Token next()
    {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_])))
        {
            ++pos_;
        }
        if (pos_ >= s_.size())
        {
            return {};
        }
        const char c = s_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.')
        {
            const char* start = s_.c_str() + pos_;
            char* end = nullptr;
            const double v = std::strtod(start, &end);
            if (end == start)
            {
                throw GrammarError("bad number at column " + std::to_string(pos_)
                                   + " in \"" + s_ + "\"");
            }
            // parity with the Python tokenizer: decimal literals only
            // (strtod would otherwise accept hex floats like 0x1A)
            for (const char* p = start; p != end; ++p)
            {
                if (std::string("0123456789.eE+-").find(*p) == std::string::npos)
                {
                    throw GrammarError("illegal numeric literal at column "
                                       + std::to_string(pos_) + " in \"" + s_ + "\"");
                }
            }
            Token t;
            t.kind = Token::Kind::Num;
            t.text.assign(start, static_cast<std::size_t>(end - start));
            t.value = v;
            pos_ += static_cast<std::size_t>(end - start);
            return t;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
        {
            std::size_t end = pos_;
            while
            (
                end < s_.size()
             && (std::isalnum(static_cast<unsigned char>(s_[end])) || s_[end] == '_')
            )
            {
                ++end;
            }
            Token t;
            t.kind = Token::Kind::Name;
            t.text = s_.substr(pos_, end - pos_);
            pos_ = end;
            return t;
        }
        if (std::string("+-*/(),").find(c) != std::string::npos)
        {
            Token t;
            t.kind = Token::Kind::Op;
            t.text = std::string(1, c);
            ++pos_;
            return t;
        }
        throw GrammarError("illegal character '" + std::string(1, c)
                           + "' at column " + std::to_string(pos_)
                           + " in \"" + s_ + "\"");
    }
};

class Parser
{
    Lexer lexer_;
    Token tok_;
    const std::string& source_;
    const bool allowConstants_;

    void advance() { tok_ = lexer_.next(); }

    bool isOp(const char* op) const
    {
        return tok_.kind == Token::Kind::Op && tok_.text == op;
    }

    void expectOp(const char* op)
    {
        if (!isOp(op))
        {
            throw GrammarError("expected '" + std::string(op) + "', got '"
                               + tok_.text + "' in \"" + source_ + "\"");
        }
        advance();
    }

    std::unique_ptr<ExprNode> makeBinary
    (
        ExprNode::Kind k,
        std::unique_ptr<ExprNode> a,
        std::unique_ptr<ExprNode> b
    )
    {
        auto n = std::make_unique<ExprNode>(k);
        n->a = std::move(a);
        n->b = std::move(b);
        return n;
    }

    std::unique_ptr<ExprNode> expr()
    {
        auto node = term();
        while (isOp("+") || isOp("-"))
        {
            const bool add = tok_.text == "+";
            advance();
            node = makeBinary
            (
                add ? ExprNode::Kind::Add : ExprNode::Kind::Sub,
                std::move(node),
                term()
            );
        }
        return node;
    }

    std::unique_ptr<ExprNode> term()
    {
        auto node = unary();
        while (isOp("*") || isOp("/"))
        {
            const bool mul = tok_.text == "*";
            advance();
            node = makeBinary
            (
                mul ? ExprNode::Kind::Mul : ExprNode::Kind::Div,
                std::move(node),
                unary()
            );
        }
        return node;
    }

    std::unique_ptr<ExprNode> unary()
    {
        if (isOp("-"))
        {
            advance();
            auto n = std::make_unique<ExprNode>(ExprNode::Kind::Neg);
            n->a = unary();
            return n;
        }
        return primary();
    }

    std::unique_ptr<ExprNode> call1(ExprNode::Kind k)
    {
        expectOp("(");
        auto n = std::make_unique<ExprNode>(k);
        n->a = expr();
        expectOp(")");
        return n;
    }

    std::unique_ptr<ExprNode> call2(ExprNode::Kind k)
    {
        expectOp("(");
        auto n = std::make_unique<ExprNode>(k);
        n->a = expr();
        expectOp(",");
        n->b = expr();
        expectOp(")");
        return n;
    }

    std::unique_ptr<ExprNode> primary()
    {
        if (tok_.kind == Token::Kind::Num)
        {
            auto n = std::make_unique<ExprNode>(ExprNode::Kind::Num);
            n->value = tok_.value;
            advance();
            return n;
        }
        if (isOp("("))
        {
            advance();
            auto n = expr();
            expectOp(")");
            return n;
        }
        if (tok_.kind == Token::Kind::Name)
        {
            const std::string name = tok_.text;
            advance();
            for (std::size_t i = 0; i < stateVars.size(); ++i)
            {
                if (name == stateVars[i])
                {
                    auto n = std::make_unique<ExprNode>(ExprNode::Kind::Var);
                    n->varIndex = i;
                    return n;
                }
            }
            if
            (
                allowConstants_
             && name.size() == 2
             && name[0] == 'c'
             && name[1] >= '0'
             && name[1] <= '7'
            )
            {
                auto n = std::make_unique<ExprNode>(ExprNode::Kind::Var);
                n->varIndex = N_STATE_VARS + static_cast<std::size_t>(name[1] - '0');
                return n;
            }
            if (name == "tanh") return call1(ExprNode::Kind::Tanh);
            if (name == "exp")  return call1(ExprNode::Kind::Exp);
            if (name == "sqrt") return call1(ExprNode::Kind::Sqrt);
            if (name == "abs")  return call1(ExprNode::Kind::Abs);
            if (name == "min")  return call2(ExprNode::Kind::Min);
            if (name == "max")  return call2(ExprNode::Kind::Max);
            throw GrammarError("unknown identifier '" + name + "' in \""
                               + source_ + "\"");
        }
        throw GrammarError("unexpected token '" + tok_.text + "' in \""
                           + source_ + "\"");
    }

public:
    Parser(const std::string& s, bool allowConstants)
    :
        lexer_(s),
        source_(s),
        allowConstants_(allowConstants)
    {
        advance();
    }

    std::unique_ptr<ExprNode> parse()
    {
        auto node = expr();
        if (tok_.kind != Token::Kind::End)
        {
            throw GrammarError("trailing tokens after expression in \""
                               + source_ + "\"");
        }
        return node;
    }
};

} // anonymous namespace

std::unique_ptr<ExprNode> parse(const std::string& source, bool allowFreeConstants)
{
    bool blank = true;
    for (const char c : source)
    {
        if (!std::isspace(static_cast<unsigned char>(c)))
        {
            blank = false;
            break;
        }
    }
    if (blank)
    {
        throw GrammarError("empty expression");
    }
    return Parser(source, allowFreeConstants).parse();
}

} // namespace expr
} // namespace tedp
