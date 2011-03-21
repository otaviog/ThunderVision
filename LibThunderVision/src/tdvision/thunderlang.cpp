#include "parserexception.hpp"
#include "thunderlang.hpp"
#include "ThunderLangLexer.h"
#include "ThunderLangParser.h"
#include <map>


TDV_NAMESPACE_BEGIN

static std::map<pANTLR3_PARSER, ThunderLangParser*> g_antlrParserMap;

static void antlDisplayRecongnitionError(pANTLR3_BASE_RECOGNIZER recognizer,
                                         pANTLR3_UINT8 * tokenNames)
{
    pANTLR3_PARSER			parser;
    pANTLR3_TREE_PARSER     tparser;
    pANTLR3_INT_STREAM      is;
    pANTLR3_STRING			ttext;
    pANTLR3_STRING			ftext;
    pANTLR3_EXCEPTION       ex;
    pANTLR3_COMMON_TOKEN    theToken;
    pANTLR3_BASE_TREE       theBaseTree;
    pANTLR3_COMMON_TREE     theCommonTree;

    assert(recognizer->type == ANTLR3_TYPE_PARSER);
    parser      = (pANTLR3_PARSER) (recognizer->super);

    ThunderLangParser *self = g_antlrParserMap[parser];
    assert(self != NULL);

    ThunderLangParser::Error error;
    // Retrieve some info for easy reading.
    //
    ex      =		recognizer->state->exception;
    ttext   =		NULL;

    // See if there is a 'filename' we can use
    //
    if	(ex->streamName == NULL)
    {
        strncpy(error.filename, "stdin", 
                ThunderLangParser::Error::MAXFILENAME);
    }
    else
    {
        ftext = ex->streamName->to8(ex->streamName);
        strncpy(error.filename, reinterpret_cast<char*>(ftext->chars),
                ThunderLangParser::Error::MAXFILENAME);
    }

    error.linenum = recognizer->state->exception->line;
    error.column = recognizer->state->exception->charPositionInLine;

    tparser     = NULL;
    is			= parser->tstream->istream;
    theToken    = (pANTLR3_COMMON_TOKEN)(recognizer->state->exception->token);
    ttext       = theToken->toString(theToken);

    const char *extraInfo = "";
    if  (theToken != NULL)
    {
        extraInfo = "at <EOF>";

        if (theToken->type != ANTLR3_TOKEN_EOF)
        {
            extraInfo = ttext == NULL
                ? "<no text for the token>"
                : reinterpret_cast<const char*>(ttext->chars);
        }
#ifdef _MSC_VER
        sprintf_s(error.description, ThunderLangParser::Error::MAXDESCRIPTION,
                  "%s, near %s",
                  reinterpret_cast<const char*>(
                      recognizer->state->exception->message),
                  extraInfo);
#else
        snprintf(error.description, ThunderLangParser::Error::MAXDESCRIPTION,
                 "%s, near %s",
                 reinterpret_cast<const char*>(
                     recognizer->state->exception->message),
                 extraInfo);
#endif
    }
    else
    {
        strncpy(error.description,
                reinterpret_cast<const char*>(
                    recognizer->state->exception->message),
                ThunderLangParser::Error::MAXDESCRIPTION);
    }
    self->___private_ADD_ERROR(error);
}

void ThunderLangParser::parseFile(const std::string &filename)
{
    pANTLR3_INPUT_STREAM input;
    pANTLR3_COMMON_TOKEN_STREAM tstream;

    pThunderLangLexer lexer;
    pThunderLangParser parser;

    //m_errors.clear();
    input = antlr3AsciiFileStreamNew( (pANTLR3_UINT8) filename.c_str());
    if ( NULL == input )
        throw ParserException("Filename doens't exists");

    lexer = ThunderLangLexerNew(input);
    if ( NULL == lexer ) {
        input->close(input);
        throw ParserException("Can't open lexer");
    }

    tstream = antlr3CommonTokenStreamSourceNew(
        ANTLR3_SIZE_HINT, TOKENSOURCE(lexer));
    if ( NULL == tstream ) {
        lexer->free(lexer);
        input->close(input);
        throw ParserException("Cant open file");
    }

    parser = ThunderLangParserNew(tstream);
    if ( NULL == parser ) {
        tstream->free(tstream);
        lexer->free(lexer);
        input->close(input);
        throw ParserException("Cant open file");
    }

    g_antlrParserMap[parser->pParser] = this;
    parser->pParser->rec->displayRecognitionError =
        antlDisplayRecongnitionError;

    parser->thunderLang(parser, this);
    g_antlrParserMap.erase(parser->pParser);

    parser->free(parser);
    tstream->free(tstream);
    lexer->free(lexer);
    input->close(input);
}

TDV_NAMESPACE_END
