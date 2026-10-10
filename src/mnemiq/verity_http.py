"""**M106.** The one way mnemiq calls Verity: a redirect is refused, never followed.

urllib follows a 3xx by rebuilding the request with every header but the content ones
(`HTTPRedirectHandler.redirect_request`), so a records pull or a trace post answered with a
redirect carried its `Authorization: Bearer` token to whatever host the redirect named. A token
request answered with one was re-sent there as a GET -- without the client secret, which rides in
the body urllib drops -- and its token was taken from a host that is not the configured issuer.
The OpenAI-side clients already refuse redirects (`follow_redirects=False`); these calls now do too.
"""

import urllib.error
import urllib.request


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code,
            f"redirect to {newurl} refused -- Verity calls follow no redirect; configure the "
            "final URL",
            headers, fp)


_OPENER = urllib.request.build_opener(_RefuseRedirect)


def urlopen(request: urllib.request.Request, *, timeout: float):
    """`urllib.request.urlopen` that refuses redirects: the seam every Verity call goes through,
    and the one tests replace."""
    return _OPENER.open(request, timeout=timeout)
